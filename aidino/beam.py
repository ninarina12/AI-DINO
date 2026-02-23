"""
Beam class for BCDI simulations.

Provides beam profile generation with support for different beam shapes,
penetration depth, and material properties.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
import torch

from torch import Tensor
from typing import Tuple, Optional, Dict

from aidino.xray_utils import wavelength_to_energy
from aidino.plot_utils import format_axis, truncate_colormap, props, create_figure_with_colorbar

class Beam:
    """
    Base class for X-ray beam profiles.
    
    All beam classes should implement the _calculate_transverse_profile method.
    """
    
    def __init__(
        self,
        wavelength: float,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize beam with fundamental properties.
        
        Parameters
        ----------
        wavelength : float
            X-ray wavelength in meters
        dtype : torch.dtype
            PyTorch data type
        device : str
            PyTorch device ('cuda' or 'cpu')
        """
        self.wavelength = wavelength
        self.dtype = dtype
        self.device = device
        
        # Calculate energy from wavelength
        self.energy = wavelength_to_energy(wavelength)
    
    def _calculate_transverse_profile(
        self,
        coords_transverse: Tensor,
        beam_center_transverse: Tuple[float, float]
    ) -> Tensor:
        """
        Calculate the beam intensity profile in the transverse plane.
        
        Parameters
        ----------
        coords_transverse : torch.Tensor
            Transverse coordinates [2, n1, n2, n3] in meters
        beam_center_transverse : tuple
            Beam center in transverse plane (meters)
        
        Returns
        -------
        torch.Tensor
            Transverse intensity profile [n1, n2, n3], normalized to max=1
        """
        raise NotImplementedError
    
    def create_profile(
        self,
        crystal,
        supercell_size: Tuple[int, int, int],
        k_i: Tensor,
        beam_center: Optional[Tuple[float, float, float]] = None,
    ) -> Tensor:
        """
        Create beam profile from incident wavevector.
        
        Parameters
        ----------
        crystal : Crystal
            Crystal object with lattice vectors
        supercell_size : tuple
            Size of supercells (s1, s2, s3)
        k_i : Tensor
            Incident wavevector (normalized or not)
        beam_center : tuple, optional
            Beam entry point in unit-cell coordinates (a, b, c indices).
            If None, automatically determined from k_i.
        """
        
        c1, c2, c3 = crystal.crystal_size
        s1, s2, s3 = supercell_size
        n1, n2, n3 = c1 // s1, c2 // s2, c3 // s3
        
        # Normalize beam direction (in lab frame)
        k_hat = k_i / torch.norm(k_i)
        k_hat = k_hat.to(device=self.device, dtype=self.dtype)
        
        # Determine entry point in lab frame, then project to crystal frame
        if beam_center is None:
            # Find which lab-frame face the beam enters from
            # The beam enters the face most perpendicular to -k_hat (i.e., facing the beam)

            # Crystal face normals in lab frame are just the normalized lattice vectors
            lattice_vectors_normalized = crystal.lattice_vectors / torch.norm(
                crystal.lattice_vectors, dim=1, keepdim=True
            )
            
            # For each of the 6 crystal faces (±a, ±b, ±c), find which is most
            # aligned with the incoming beam direction (-k_hat = beam comes from there)
            face_normals_crystal = torch.cat([
                 lattice_vectors_normalized,
                -lattice_vectors_normalized
            ], dim=0)  # [6, 3] in lab frame
            
            alignments = torch.matmul(face_normals_crystal, -k_hat)
            best_face_idx = torch.argmax(alignments).item()
            
            # Decode: indices 0,1,2 = +a,+b,+c face; 3,4,5 = -a,-b,-c face
            dominant_crystal_axis = best_face_idx % 3
            enters_from_positive = best_face_idx < 3
            
            beam_center = [c1 / 2.0, c2 / 2.0, c3 / 2.0]
            
            if enters_from_positive:
                # Beam enters from high-index face
                beam_center[dominant_crystal_axis] = (
                    crystal.crystal_size[dominant_crystal_axis] -
                    supercell_size[dominant_crystal_axis] / 2.0
                )
            else:
                # Beam enters from low-index face
                beam_center[dominant_crystal_axis] = supercell_size[dominant_crystal_axis] / 2.0
            
            axis_name = ['a', 'b', 'c'][dominant_crystal_axis]
            side = '+' if enters_from_positive else '-'
            print(f"Auto beam entry: crystal {side}{axis_name} face, "
                  f"\nBeam_center: {tuple(beam_center)}")
        
        # Create coordinate grids in crystal frame
        i = torch.arange(n1, device=self.device, dtype=self.dtype)
        j = torch.arange(n2, device=self.device, dtype=self.dtype)
        k = torch.arange(n3, device=self.device, dtype=self.dtype)
        
        ii, jj, kk = torch.meshgrid(i, j, k, indexing="ij")
        
        # Supercell centers in unit cell coordinates
        coords_uc = torch.stack([
            ii * s1 + s1 / 2.0,  # Ranges from s1/2 to (n1-1)*s1 + s1/2 = c1 - s1/2
            jj * s2 + s2 / 2.0,
            kk * s3 + s3 / 2.0
        ], dim=0)  # [3, n1, n2, n3]
        
        # Transform to lab frame using lattice vectors
        # r_lab = n_a * a_vec + n_b * b_vec + n_c * c_vec
        coords_lab = (
            coords_uc[0].unsqueeze(0) * crystal.lattice_vectors[0].view(3, 1, 1, 1) +
            coords_uc[1].unsqueeze(0) * crystal.lattice_vectors[1].view(3, 1, 1, 1) +
            coords_uc[2].unsqueeze(0) * crystal.lattice_vectors[2].view(3, 1, 1, 1)
        )  # [3, n1, n2, n3] in lab frame (meters)
        
        # Transform beam_center to lab frame
        beam_center_lab = (
            beam_center[0] * crystal.lattice_vectors[0] +
            beam_center[1] * crystal.lattice_vectors[1] +
            beam_center[2] * crystal.lattice_vectors[2]
        )  # [3] in lab frame (meters)
        
        # Position relative to beam entry
        r = coords_lab - beam_center_lab.view(3, 1, 1, 1)
        del coords_lab  # Free this immediately
        
        # Distance along beam direction
        depth = torch.sum(r * k_hat.view(3, 1, 1, 1), dim=0)  # [n1, n2, n3]
        
        # Transverse component (perpendicular to beam)
        r_parallel = depth.unsqueeze(0) * k_hat.view(3, 1, 1, 1)
        r_transverse = r - r_parallel  # [3, n1, n2, n3]
        del r, r_parallel  # Free these immediately
        
        # Create orthonormal basis in transverse plane
        if torch.abs(k_hat[2]) < 0.9:
            arbitrary = torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype)
        else:
            arbitrary = torch.tensor([1., 0., 0.], device=self.device, dtype=self.dtype)
        
        u_transverse = arbitrary - torch.dot(arbitrary, k_hat) * k_hat
        u_transverse = u_transverse / torch.norm(u_transverse)
        
        v_transverse = torch.linalg.cross(k_hat, u_transverse)
        v_transverse = v_transverse / torch.norm(v_transverse)
        
        # Project transverse positions onto 2D basis
        coords_transverse = torch.stack([
            torch.sum(r_transverse * u_transverse.view(3, 1, 1, 1), dim=0),
            torch.sum(r_transverse * v_transverse.view(3, 1, 1, 1), dim=0)
        ], dim=0)  # [2, n1, n2, n3]
        del r_transverse  # Free this immediately
        
        # Calculate transverse profile (calls subclass-specific method)
        transverse_profile = self._calculate_transverse_profile(
            coords_transverse, 
            (0.0, 0.0)  # Beam centered at origin in transverse coords
        )
        
        # Get penetration depth
        penetration_depth = crystal.get_penetration_depth(self.wavelength)
        
        # Apply exponential attenuation along beam direction
        # Shift depth so the shallowest point in the crystal is the entry reference
        depth_from_entry = depth - depth.min()
        attenuation = torch.exp(-depth_from_entry / penetration_depth)
        
        # Combine transverse and longitudinal profiles
        profile = transverse_profile * attenuation
        
        self.profile = profile.unsqueeze(0)  # [1, n1, n2, n3]
        
    def plot_profile(
        self,
        crystal,
        supercell_size: Tuple[int, int, int],
        k_i: Tensor,
        k_f: Tensor,
        intensity_threshold: float = 0.,
        alpha_scale: float = 0.5,
        subplot_size: float = 3.5,
        wspace: float = 0.6,
        hspace: float = 0.2,
    ):
        """
        Plot calculated beam profile as a two-row figure:
          - Row 0: 3D lab-frame scatter (lab-x vertical) with k_i, k_f, Q arrows
          - Row 1: bc, ac, ab crystal-frame slice panels with shared colorbar
    
        Parameters
        ----------
        crystal : Crystal
        supercell_size : tuple
        k_i : Tensor
            Incident wavevector (lab frame).
        k_f : Tensor
            Exit wavevector (lab frame).
        intensity_threshold : float
            Voxels below this intensity omitted from 3D scatter.
        alpha_scale : float
            Maximum alpha for most intense voxel in 3D scatter.
        subplot_size : float
            Physical size of each main subplot cell in inches.
        wspace : float
            Fractional spacing between slice panels.
        hspace : float
            Fractional spacing between the two rows.
        """
    
        # Shared slice data
        beam_3d_np = self.profile.squeeze(0).cpu().numpy()
        n1, n2, n3 = beam_3d_np.shape
        s1, s2, s3 = supercell_size
    
        supercell_a_size = s1 * torch.norm(crystal.lattice_vectors[0]).item() * 1e9
        supercell_b_size = s2 * torch.norm(crystal.lattice_vectors[1]).item() * 1e9
        supercell_c_size = s3 * torch.norm(crystal.lattice_vectors[2]).item() * 1e9
    
        total_a = n1 * supercell_a_size
        total_b = n2 * supercell_b_size
        total_c = n3 * supercell_c_size
    
        slice_a = n1 // 2;  slice_b = n2 // 2;  slice_c = n3 // 2
        slice_a_pos = slice_a * supercell_a_size
        slice_b_pos = slice_b * supercell_b_size
        slice_c_pos = slice_c * supercell_c_size
    
        bc_slice = beam_3d_np[slice_a, :, :]
        ac_slice = beam_3d_np[:, slice_b, :]
        ab_slice = beam_3d_np[:, :, slice_c]
    
        vmin, vmax = 0, 1.0
    
        # Compute figsize from physical panel sizes
        cbar_ratio = 0.05
    
        ratio_bc = total_c / total_b
        ratio_ac = total_c / total_a
        ratio_ab = total_b / total_a
    
        w_bc   = subplot_size * ratio_bc
        w_ac   = subplot_size * ratio_ac
        w_ab   = subplot_size * ratio_ab
        w_cbar = subplot_size * cbar_ratio
    
        slice_gap = wspace * subplot_size
        total_w   = w_bc + w_ac + w_ab + 2 * slice_gap + w_cbar
    
        h_slice = subplot_size
        h_3d    = subplot_size * 1.5
        row_gap = hspace * (h_3d + h_slice) / 2
        total_h = h_3d + h_slice + row_gap
    
        # Build figure
        fig = plt.figure(figsize=(total_w, total_h), facecolor='white')
    
        outer = fig.add_gridspec(
            2, 1,
            height_ratios=[h_3d, h_slice],
            hspace=hspace,
        )
        ax3d = fig.add_subplot(outer[0], projection='3d')
    
        slices_w = w_bc + w_ac + w_ab + 2 * slice_gap
        gs_bot = outer[1].subgridspec(
            1, 2,
            width_ratios=[slices_w, w_cbar],
            wspace=0,
        )
        gs_slices = gs_bot[0, 0].subgridspec(
            1, 3,
            width_ratios=[ratio_bc, ratio_ac, ratio_ab],
            wspace=wspace,
        )
        ax_bc   = fig.add_subplot(gs_slices[0, 0])
        ax_ac   = fig.add_subplot(gs_slices[0, 1])
        ax_ab   = fig.add_subplot(gs_slices[0, 2])
        ax_cbar = fig.add_subplot(gs_bot[0, 1])
    
        # 3D panel
        i_ = torch.arange(n1, device=self.device, dtype=self.dtype)
        j_ = torch.arange(n2, device=self.device, dtype=self.dtype)
        k_ = torch.arange(n3, device=self.device, dtype=self.dtype)
        ii, jj, kk = torch.meshgrid(i_, j_, k_, indexing="ij")
    
        coords_uc = torch.stack([
            ii * s1 + s1 / 2.0,
            jj * s2 + s2 / 2.0,
            kk * s3 + s3 / 2.0,
        ], dim=0)
    
        coords_lab = (
            coords_uc[0].unsqueeze(0) * crystal.lattice_vectors[0].view(3,1,1,1) +
            coords_uc[1].unsqueeze(0) * crystal.lattice_vectors[1].view(3,1,1,1) +
            coords_uc[2].unsqueeze(0) * crystal.lattice_vectors[2].view(3,1,1,1)
        ) * 1e9  # nm
    
        lab_x = coords_lab[0].cpu().numpy().ravel()
        lab_y = coords_lab[1].cpu().numpy().ravel()
        lab_z = coords_lab[2].cpu().numpy().ravel()
        intensity = self.profile.squeeze(0).cpu().numpy().ravel()
    
        mask = intensity >= intensity_threshold
        lab_x = lab_x[mask];  lab_y = lab_y[mask]
        lab_z = lab_z[mask];  intensity = intensity[mask]
    
        colors = cm.lapaz(mpl.colors.Normalize(vmin=0, vmax=1)(intensity))
        colors[:, 3] = intensity * alpha_scale
    
        k_i_np = k_i.cpu().numpy(); k_i_np /= np.linalg.norm(k_i_np)
        k_f_np = k_f.cpu().numpy(); k_f_np /= np.linalg.norm(k_f_np)
    
        extent_nm = max(lab_x.max()-lab_x.min(),
                        lab_y.max()-lab_y.min(),
                        lab_z.max()-lab_z.min())
        arrow_len = extent_nm * 0.45
        center = np.array([(lab_x.max()+lab_x.min())/2,
                           (lab_y.max()+lab_y.min())/2,
                           (lab_z.max()+lab_z.min())/2])
    
        # lab-x vertical: mpl axes → (lab_y, lab_z, lab_x)
        def to_mpl(lx, ly, lz):
            return ly, lz, lx
    
        ax3d.scatter(*to_mpl(lab_x, lab_y, lab_z),
                     c=colors, s=4, linewidths=0, depthshade=False)

        ax3d.quiver(*to_mpl(*(center - k_i_np*arrow_len*1.1)),
            *to_mpl(*(k_i_np*arrow_len)),
            color=cm.vik(0.2), linewidth=2,
            arrow_length_ratio=0.15, label=r'$k_i$')
        
        ax3d.quiver(*to_mpl(*center),
                    *to_mpl(*(k_f_np*arrow_len)),
                    color=cm.vik(0.8), linewidth=2,
                    arrow_length_ratio=0.15, label=r'$k_f$')
    
        q_np = k_f_np - k_i_np
        q_np = q_np / np.linalg.norm(q_np)
        ax3d.quiver(*to_mpl(*center),
                    *to_mpl(*(q_np*arrow_len)),
                    color='black', linewidth=2,
                    arrow_length_ratio=0.15, label=r'$Q$')
        
        ax3d.legend(loc='upper left', frameon=False)
    
        format_axis(ax3d)
        ax3d.set_xlabel('y (nm)', fontproperties=props, labelpad=4)
        ax3d.set_ylabel('z (nm)', fontproperties=props, labelpad=4)
        ax3d.set_zlabel('x (nm)', fontproperties=props, labelpad=4)
        x_label_props = ax3d.get_xticklabels()[0].get_font_properties()
        for label in ax3d.get_zticklabels():
            label.set_font_properties(x_label_props)
    
        ki_start = center - k_i_np * arrow_len * 1.1
        kf_end   = center + k_f_np * arrow_len
        all_mpl = np.array([
            [lab_y.min(), lab_z.min(), lab_x.min()],
            [lab_y.max(), lab_z.max(), lab_x.max()],
            [ki_start[1], ki_start[2], ki_start[0]],
            [kf_end[1],   kf_end[2],   kf_end[0]  ],
        ])
        mins = all_mpl.min(axis=0)
        maxs = all_mpl.max(axis=0)
        half_range = (maxs - mins).max() / 2
        mids = (mins + maxs) / 2
        ax3d.set_xlim(mids[0] - half_range, mids[0] + half_range)
        ax3d.set_ylim(mids[1] - half_range, mids[1] + half_range)
        ax3d.set_zlim(mids[2] - half_range, mids[2] + half_range)

        # Lab axis indicators — computed after limits are set so they stay in view
        ax_len = extent_nm * 0.15
        # Origin is at the low corner of the actual axis limits
        # Inverse of to_mpl: lab_x=mpl_z, lab_y=mpl_x, lab_z=mpl_y
        origin_nm = np.array([
            ax3d.get_zlim()[0] + ax_len * 0.3,   # lab_x
            ax3d.get_xlim()[0] + ax_len * 0.3,   # lab_y
            ax3d.get_ylim()[0] + ax_len * 0.3,   # lab_z
        ])
        for vec, lbl, col in [
            (np.array([1,0,0]), 'x',      cm.managua(70)),
            (np.array([0,1,0]), '     y', cm.managua(30)),
            (np.array([0,0,1]), '   z',   cm.managua(200)),
        ]:
            ax3d.quiver(*to_mpl(*origin_nm), *to_mpl(*(vec*ax_len)),
                        color=col, linewidth=1.5, arrow_length_ratio=0.3)
            ax3d.text(*to_mpl(*(origin_nm + vec*ax_len*1.4)),
                      lbl, color=col, fontsize=8, fontproperties=props, ha='center')

        ax3d.view_init(elev=20, azim=-55)
        ax3d.grid(False)
        ax3d.xaxis.pane.fill = False
        ax3d.yaxis.pane.fill = False
        ax3d.zaxis.pane.fill = False
        ax3d.xaxis.pane.set_edgecolor('lightgrey')
        ax3d.yaxis.pane.set_edgecolor('lightgrey')
        ax3d.zaxis.pane.set_edgecolor('lightgrey')
    
        # Slice panels
        im0 = ax_bc.imshow(bc_slice, cmap=cm.lapaz, origin='lower', aspect='auto',
                           extent=[0, total_c, 0, total_b], vmin=vmin, vmax=vmax)
        format_axis(ax_bc,
                    xlabel='c-axis (nm)', ylabel='b-axis (nm)',
                    title=f'bc-plane (⊥ a)\nSlice at a = {slice_a_pos:.1f} nm')
    
        im1 = ax_ac.imshow(ac_slice, cmap=cm.lapaz, origin='lower', aspect='auto',
                           extent=[0, total_c, 0, total_a], vmin=vmin, vmax=vmax)
        format_axis(ax_ac,
                    xlabel='c-axis (nm)', ylabel='a-axis (nm)',
                    title=f'ac-plane (⊥ b)\nSlice at b = {slice_b_pos:.1f} nm')
    
        im2 = ax_ab.imshow(ab_slice, cmap=cm.lapaz, origin='lower', aspect='auto',
                           extent=[0, total_b, 0, total_a], vmin=vmin, vmax=vmax)
        format_axis(ax_ab,
                    xlabel='b-axis (nm)', ylabel='a-axis (nm)',
                    title=f'ab-plane (⊥ c)\nSlice at c = {slice_c_pos:.1f} nm')
    
        cbar = fig.colorbar(im2, cax=ax_cbar)
        cbar.set_label('Beam Intensity', fontproperties=props)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(props)
    
        ax_bc.set_aspect('equal')
        ax_ac.set_aspect('equal')
        ax_ab.set_aspect('equal')
    
        return fig
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(wavelength={self.wavelength:.3e}m, "
                f"energy={self.energy / 1000.:.2f}keV)")


class GaussianBeam(Beam):
    """Gaussian beam profile."""
    
    def __init__(
        self,
        wavelength: float,
        fwhm: float,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize Gaussian beam.
        
        Parameters
        ----------
        wavelength : float
            X-ray wavelength in meters
        fwhm : float
            Full width at half maximum in meters
        dtype : torch.dtype
            PyTorch data type
        device : str
            PyTorch device
        """
        super().__init__(wavelength, dtype, device)
        self.fwhm = fwhm
        
        # Convert FWHM to Gaussian sigma
        self.sigma = fwhm / (2 * torch.sqrt(2 * torch.log(torch.tensor(2.0))))
    
    def _calculate_transverse_profile(
        self,
        coords_transverse: Tensor,
        beam_center_transverse: Tuple[float, float]
    ) -> Tensor:
        """Calculate Gaussian profile in transverse plane."""
        
        # Calculate distance from beam center
        r_sq = torch.zeros_like(coords_transverse[0])
        for i, center in enumerate(beam_center_transverse):
            r_sq += (coords_transverse[i] - center) ** 2
        
        # Gaussian profile
        profile = torch.exp(-r_sq / (2 * self.sigma ** 2))
        
        return profile
    
    def __repr__(self):
        return (f"GaussianBeam(wavelength={self.wavelength:.3e}m, "
                f"fwhm={self.fwhm:.3e}m, energy={self.energy / 1000.:.2f}keV)")


class EllipticalBeam(Beam):
    """Elliptical Gaussian beam (different FWHM in two transverse directions)."""
    
    def __init__(
        self,
        wavelength: float,
        fwhm_horizontal: float,
        fwhm_vertical: float,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize elliptical Gaussian beam.
        
        Parameters
        ----------
        wavelength : float
            X-ray wavelength in meters
        fwhm_horizontal : float
            FWHM in first transverse direction (meters)
        fwhm_vertical : float
            FWHM in second transverse direction (meters)
        dtype : torch.dtype
            PyTorch data type
        device : str
            PyTorch device
        """
        super().__init__(wavelength, dtype, device)
        self.fwhm_horizontal = fwhm_horizontal
        self.fwhm_vertical = fwhm_vertical
        
        # Convert to sigmas
        sqrt_2ln2 = torch.sqrt(2 * torch.log(torch.tensor(2.0)))
        self.sigma_h = fwhm_horizontal / (2 * sqrt_2ln2)
        self.sigma_v = fwhm_vertical / (2 * sqrt_2ln2)
    
    def _calculate_transverse_profile(
        self,
        coords_transverse: Tensor,
        beam_center_transverse: Tuple[float, float]
    ) -> Tensor:
        """Calculate elliptical Gaussian profile."""
        
        # Distance in each direction, normalized by respective sigma
        delta_h = (coords_transverse[0] - beam_center_transverse[0]) / self.sigma_h
        delta_v = (coords_transverse[1] - beam_center_transverse[1]) / self.sigma_v
        
        profile = torch.exp(-0.5 * (delta_h**2 + delta_v**2))
        
        return profile
    
    def __repr__(self):
        return (f"EllipticalBeam(wavelength={self.wavelength:.3e}m, "
                f"fwhm_h={self.fwhm_horizontal:.3e}m, fwhm_v={self.fwhm_vertical:.3e}m, "
                f"energy={self.energy / 1000.:.2f}keV)")


class TopHatBeam(Beam):
    """Top-hat (flat) beam profile with sharp cutoff."""
    
    def __init__(
        self,
        wavelength: float,
        diameter: float,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize top-hat beam.
        
        Parameters
        ----------
        wavelength : float
            X-ray wavelength in meters
        diameter : float
            Beam diameter in meters
        dtype : torch.dtype
            PyTorch data type
        device : str
            PyTorch device
        """
        super().__init__(wavelength, dtype, device)
        self.diameter = diameter
        self.radius = diameter / 2.0
    
    def _calculate_transverse_profile(
        self,
        coords_transverse: Tensor,
        beam_center_transverse: Tuple[float, float]
    ) -> Tensor:
        """Calculate top-hat profile (1 inside radius, 0 outside)."""
        
        # Calculate distance from beam center
        r_sq = torch.zeros_like(coords_transverse[0])
        for i, center in enumerate(beam_center_transverse):
            r_sq += (coords_transverse[i] - center) ** 2
        
        r = torch.sqrt(r_sq)
        
        # Step function
        profile = (r <= self.radius).to(self.dtype)
        
        return profile
    
    def __repr__(self):
        return (f"TopHatBeam(wavelength={self.wavelength:.3e}m, "
                f"diameter={self.diameter:.3e}m, energy={self.energy / 1000.:.2f}keV)")


class CustomBeam(Beam):
    """
    Custom beam profile from measured or simulated data.
    
    Useful for incorporating actual measured beam profiles from beamline characterization.
    """
    
    def __init__(
        self,
        wavelength: float,
        profile_func,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize custom beam from a function.
        
        Parameters
        ----------
        wavelength : float
            X-ray wavelength in meters
        profile_func : callable
            Function that takes (coords_transverse, beam_center_transverse) and
            returns intensity profile
        dtype : torch.dtype
            PyTorch data type
        device : str
            PyTorch device
        """
        super().__init__(wavelength, dtype, device)
        self.profile_func = profile_func
    
    def _calculate_transverse_profile(
        self,
        coords_transverse: Tensor,
        beam_center_transverse: Tuple[float, float]
    ) -> Tensor:
        """Calculate custom profile using provided function."""
        return self.profile_func(coords_transverse, beam_center_transverse)
    
    def __repr__(self):
        return (f"CustomBeam(wavelength={self.wavelength:.3e}m, "
                f"energy={self.energy / 1000.:.2f}keV, profile=custom)")