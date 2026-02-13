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
            
            # Define the 6 possible entry faces (±x, ±y, ±z in lab frame)
            face_normals = torch.tensor([
                [-1., 0., 0.],  # -x face (beam coming from +x)
                [1., 0., 0.],   # +x face (beam coming from -x)
                [0., -1., 0.],  # -y face (beam coming from +y)
                [0., 1., 0.],   # +y face (beam coming from -y)
                [0., 0., -1.],  # -z face (beam coming from +z)
                [0., 0., 1.],   # +z face (beam coming from -z)
            ], device=self.device, dtype=self.dtype)
            
            # Find the face most aligned with -k_hat (beam enters this face)
            # dot(normal, k_hat) > 0 means beam is entering through this face
            alignments = torch.matmul(face_normals, -k_hat)
            best_face_idx = torch.argmax(alignments).item()
            entry_normal = face_normals[best_face_idx]
            
            # Map face index to description
            face_names = ['-x', '+x', '-y', '+y', '-z', '+z']
            entry_face_name = face_names[best_face_idx]
            
            # Default: center in all crystal dimensions
            beam_center = [c1 / 2.0, c2 / 2.0, c3 / 2.0]
            
            # Project the entry face normal onto crystal lattice vectors
            # to find which crystal axis corresponds to this lab-frame face
            lattice_vectors_normalized = crystal.lattice_vectors / torch.norm(
                crystal.lattice_vectors, dim=1, keepdim=True
            )
            
            # For each crystal axis, check alignment with entry face normal
            projections = torch.stack([
                torch.dot(entry_normal, lattice_vectors_normalized[0]),
                torch.dot(entry_normal, lattice_vectors_normalized[1]),
                torch.dot(entry_normal, lattice_vectors_normalized[2])
            ])
            
            # Find which crystal axis is most perpendicular to entry face
            projections_abs = torch.abs(projections)
            dominant_crystal_axis = torch.argmax(projections_abs).item()
            projection_sign = projections[dominant_crystal_axis].item()
            
            # Set entry position along the dominant crystal axis
            # If projection_sign > 0: normal points in -crystal_axis direction
            #   → beam enters from high crystal indices
            # If projection_sign < 0: normal points in +crystal_axis direction
            #   → beam enters from low crystal indices
            if projection_sign > 0:
                # Beam enters from high-index face
                beam_center[dominant_crystal_axis] = (
                    crystal.crystal_size[dominant_crystal_axis] - 
                    supercell_size[dominant_crystal_axis] / 2.0
                )
            else:
                # Beam enters from low-index face
                beam_center[dominant_crystal_axis] = supercell_size[dominant_crystal_axis] / 2.0
                
            print(f"Auto beam entry: {entry_face_name} face in lab frame "
                  f"→ crystal axis {['a','b','c'][dominant_crystal_axis]} "
                  f"({'+'if projection_sign>0 else '-'} direction), "
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
        # Only attenuate where beam has entered (depth >= 0)
        attenuation = torch.exp(-torch.clamp(depth, min=0) / penetration_depth)
        
        # Zero out regions before beam entry (negative depth)
        attenuation = torch.where(depth >= 0, attenuation, torch.zeros_like(attenuation))
        
        # Combine transverse and longitudinal profiles
        profile = transverse_profile * attenuation
        
        self.profile = profile.unsqueeze(0)  # [1, n1, n2, n3]

    def plot_profile(
        self,
        crystal,
        supercell_size: Tuple[int, int, int],
    ) -> Tensor:
        """
        Plot calculated beam profile.
        
        Parameters
        ----------
        crystal : Crystal
            Crystal object with lattice vectors
        supercell_size : tuple
            Size of supercells (s1, s2, s3)
        """
        
        # Get the 3D beam profile (remove batch dimension)
        beam_3d = self.profile.squeeze(0).cpu().numpy()
        n1, n2, n3 = beam_3d.shape
        s1, s2, s3 = supercell_size
        
        # Calculate supercell sizes in physical units (nm)
        supercell_a_size = s1 * torch.norm(crystal.lattice_vectors[0]).item() * 1e9  # nm
        supercell_b_size = s2 * torch.norm(crystal.lattice_vectors[1]).item() * 1e9  # nm
        supercell_c_size = s3 * torch.norm(crystal.lattice_vectors[2]).item() * 1e9  # nm
        
        # Total dimensions in physical units
        total_a = n1 * supercell_a_size
        total_b = n2 * supercell_b_size
        total_c = n3 * supercell_c_size
        
        # Define slice positions (at center)
        slice_a = n1 // 2
        slice_b = n2 // 2
        slice_c = n3 // 2
        
        # Calculate physical position of slices
        slice_a_pos = slice_a * supercell_a_size
        slice_b_pos = slice_b * supercell_b_size
        slice_c_pos = slice_c * supercell_c_size
        
        # Get the three unique slices
        bc_slice = beam_3d[slice_a, :, :]  # Perpendicular to a-axis
        ac_slice = beam_3d[:, slice_b, :]  # Perpendicular to b-axis
        ab_slice = beam_3d[:, :, slice_c]  # Perpendicular to c-axis
        
        # Create figure with shared colorbar
        fig, axes = create_figure_with_colorbar(
            num_rows=1, 
            num_cols=4,
            wspace=0.25
        )
        
        # Get global min/max for shared colorbar
        vmin = 0
        vmax = 1.
        
        # Plot bc-slice (perpendicular to a-axis)
        im0 = axes[0].imshow(bc_slice, cmap=cm.lapaz, origin='lower', aspect='auto',
                             extent=[0, total_c, 0, total_b], vmin=vmin, vmax=vmax)
        format_axis(axes[0], 
                   xlabel='c-axis (nm)', 
                   ylabel='b-axis (nm)',
                   title=f'bc-plane (⊥ a)\nSlice at a = {slice_a_pos:.1f} nm')
        
        # Plot ac-slice (perpendicular to b-axis)
        im1 = axes[1].imshow(ac_slice, cmap=cm.lapaz, origin='lower', aspect='auto',
                             extent=[0, total_c, 0, total_a], vmin=vmin, vmax=vmax)
        format_axis(axes[1],
                   xlabel='c-axis (nm)',
                   ylabel='a-axis (nm)',
                   title=f'ac-plane (⊥ b)\nSlice at b = {slice_b_pos:.1f} nm')
        
        # Plot ab-slice (perpendicular to c-axis)
        im2 = axes[2].imshow(ab_slice, cmap=cm.lapaz, origin='lower', aspect='auto',
                             extent=[0, total_b, 0, total_a], vmin=vmin, vmax=vmax)
        format_axis(axes[2],
                   xlabel='b-axis (nm)',
                   ylabel='a-axis (nm)',
                   title=f'ab-plane (⊥ c)\nSlice at c = {slice_c_pos:.1f} nm')
        
        # Add shared colorbar in the 4th column
        cbar = fig.colorbar(im2, cax=axes[3])
        cbar.set_label('Beam Intensity', fontproperties=props)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(props)
        
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
        axes[2].set_aspect('equal')
        
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