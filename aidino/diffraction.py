import torch

from typing import Tuple, List, Dict, Optional, Union
from torch import Tensor

from aidino.sample import Crystal

class BraggCoherentDiffraction:
    """
    Class for simulating Bragg coherent diffraction.
    """
    
    def __init__(
        self,
        crystal: Crystal,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize a Bragg coherent diffraction simulator.
        
        Parameters:
        -----------
        crystal: Crystal
            An instance of a Crystal object
        dtype: torch.dtype
            torch data type
        device: str
            torch device ('cuda' or 'cpu')
        """
        self.crystal = crystal
        self.dtype = dtype
        self.device = device
        
    def calculate_structure_factor(self, q_vectors: Tensor) -> Tensor:
        """
        Calculate the structure factor for a given set of q-vectors.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
            
        Returns:
        --------
        torch.Tensor
            Structure factor as a complex tensor of shape [...]
        """
        
        # Store original shape of q_vectors for later reshaping
        q_size_original = q_vectors.shape[:-1]
        
        # Reshape q_vectors to [n_pixels, 3] for matrix multiplication
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]
        
        # Calculate q·r for each atom in the unit cell and each q-vector
        # q_vectors_flat shape: [n_pixels, 3]
        # atom_positions shape: [n_atoms, 3]
        # Result shape: [n_pixels, n_atoms]
        q_dot_r = torch.matmul(q_vectors_flat, self.crystal.atom_positions.T)
        
        # Calculate e^(-iq·r) for each atom and each q-vector
        phase_factors = torch.exp(-1j * q_dot_r)

        # Calculate |q| for form factor (convert to Å for typical form factor formulas)
        # Result shape: [n_pixels, 1]
        q_magnitude = torch.norm(q_vectors_flat, dim=-1, keepdim=True) * 1e-10

        # Vectorized calculation of form factors for all atoms and all q values
        form_factors = self.crystal.calculate_form_factors(q_magnitude)
        
        # Multiply by form factors and sum over atoms
        # Result shape: [n_pixels]
        structure_factor = torch.sum(form_factors * phase_factors, dim=-1)
        
        # Reshape back to original q_vectors shape
        structure_factor = structure_factor.view(q_size_original)
        
        return structure_factor

    def calculate_supercell_scattering(
        self,
        q_vectors: Tensor,
        supercell_size: Tuple[int, int, int],
        mask: Optional[Tensor] = None,
        q_batch_size: Optional[int] = None) -> Tensor:
        
        """
        Calculate scattering using the supercell approach from Mokhtar et al.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
        supercell_size: Tuple
            Size of supercells (d1, d2, d3) in unit cells
        mask: torch.Tensor, optional
            Optional mask of shape [batch_size, n1, n2, n3] or [batch_size, n_sc1, n_sc2, n_sc3]
        q_batch_size: int, optional
            Number of q-vectors to process at once. If None, process all at once.

        Returns:
        --------
        torch.Tensor
            Scattering amplitude as a complex tensor
        """

        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size
        
        # Generate supercell indices
        i_indices = torch.arange(0, n1, d1, dtype=self.dtype, device=self.device)
        j_indices = torch.arange(0, n2, d2, dtype=self.dtype, device=self.device)
        k_indices = torch.arange(0, n3, d3, dtype=self.dtype, device=self.device)
        
        # Compute a grid of all indices
        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='ij')

        # Reshape to [n_supercells, 3]
        supercell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=-1)
        
        # Calculate positions in real space
        # Result shape: [n_supercells, 3]
        supercell_positions = torch.matmul(supercell_indices, self.crystal.lattice_vectors)
        
        # Store original shape of q_vectors for later reshaping
        q_size_original = q_vectors.shape[:-1]
        
        # Reshape q_vectors to [n_pixels, 3] for matrix multiplication
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]
    
        # Determine q_vector batch size
        if q_batch_size is None:
            q_batch_size = n_pixels
        
        # Process scattering calculations in q_vector batches
        scattering_results = []

        for i in range(0, n_pixels, q_batch_size):
            q_batch = q_vectors_flat[i:i+q_batch_size]
    
            # Calculate q·R for each supercell and each q-vector
            # Result shape: [n_pixels, n_supercells]
            q_dot_R = torch.matmul(q_batch, supercell_positions.T)
    
            # Calculate e^(-iq·R) for each supercell and each q-vector
            phase_factors = torch.exp(-1j * q_dot_R)
    
            if mask is not None:
                n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3
                
                # Convert mask to supercell level if needed
                if mask.shape[-3:] == (n1, n2, n3):
                    # Mask is at unit cell resolution - need to downsample to supercell resolution
                    # Reshape mask to group unit cells into supercells
                    mask_grouped = mask.view(-1, n_sc1, d1, n_sc2, d2, n_sc3, d3)
    
                    # Average over supercell dimensions
                    supercell_mask = torch.mean(mask_grouped, dim=(2, 4, 6))
                    
                elif mask.shape[-3:] == (n_sc1, n_sc2, n_sc3):
                    # Mask is already at supercell resolution
                    supercell_mask = mask
                    
                else:
                    raise ValueError(
                        f"Mask shape {mask.shape[-3:]} is incompatible with crystal grid "
                        f"of size ({n1}, {n2}, {n3}) or supercell grid ({n_sc1}, {n_sc2}, {n_sc3})"
                    )
                
                # Flatten and apply mask to phase factors
                supercell_mask_flat = supercell_mask.flatten(start_dim=1)  # [batch_size, n_supercells]
                phase_factors = phase_factors.unsqueeze(0) * supercell_mask_flat.unsqueeze(1)  # [batch_size, n_pixels, n_supercells]
            
            # Calculate structure factor for the unit cell and add a batch dimension
            S_q = self.calculate_structure_factor(q_batch).unsqueeze(0)
            
            # Number of unit cells per supercell
            cells_per_supercell = d1 * d2 * d3
            
            # Multiply structure factor by sum of phase factors and scale by supercell size
            # Result shape: [batch_size, n_pixels]
            scattering_batch = S_q * torch.sum(phase_factors, dim=-1) * cells_per_supercell
    
            # Add global position phase shift
            # Calculate q·R_g for each q-vector
            # grain_position shape: [3]
            # q_vectors_flat shape: [n_pixels, 3]
            # Result shape: [n_pixels]
            q_dot_global_position = torch.matmul(q_batch, self.crystal.position)
            
            # Calculate e^(-iq·R_g) for each q-vector
            global_phase_factor = torch.exp(-1j * q_dot_global_position)
            
            # Apply global position phase shift
            scattering_batch = scattering_batch * global_phase_factor.unsqueeze(0)

            # Append scattering results
            scattering_results.append(scattering_batch)

        # Concatenate along q-vector dimension
        scattering_amplitude = torch.cat(scattering_results, dim=1)
    
        # Reshape back to original q_vector shape
        scattering_amplitude = scattering_amplitude.view(-1, *q_size_original)
        
        return scattering_amplitude

    def calculate_supercell_scattering_with_displacements(
        self,
        q_vectors: Tensor,
        supercell_size: Tuple[int, int, int],
        displacement_field: Tensor,
        mask: Optional[Tensor] = None,
        q_batch_size: Optional[int] = None
    ) -> Tensor:
        """
        Calculate scattering using the supercell approach from Mokhtar et al. with averaged displacements for each supercell.
        
        Parameters:
        -----------
        q_vectors: torch.Tensor
            Tensor of shape [..., 3] containing q-vectors
        supercell_size: Tuple
            Size of supercells (d1, d2, d3) in unit cells
        displacement_field: torch.Tensor
            Tensor of shape [batch_size, n1, n2, n3, n_atoms, 3] or [batch_size, n_sc1, n_sc2, n_sc3, n_atoms, 3]
            containing the unique or supercell-averaged displacement field for each atom, respectively
        mask: torch.Tensor, optional
            Optional mask of shape [batch_size, n1, n2, n3] or [batch_size, n_sc1, n_sc2, n_sc3]
        q_batch_size: int, optional
            Number of q-vectors to process at once. If None, process all at once.
            
        Returns:
        --------
        torch.Tensor
            Scattering amplitude as a complex tensor
        """
        
        d1, d2, d3 = supercell_size
        n1, n2, n3 = self.crystal.crystal_size
        
        # Check if dimensions divide evenly
        assert (n1 % d1 == 0) and (n2 % d2 == 0) and (n3 % d3 == 0)

        n_sc1, n_sc2, n_sc3 = n1 // d1, n2 // d2, n3 // d3
        
        # Generate supercell indices
        i_indices = torch.arange(0, n1, d1, dtype=self.dtype, device=self.device)
        j_indices = torch.arange(0, n2, d2, dtype=self.dtype, device=self.device)
        k_indices = torch.arange(0, n3, d3, dtype=self.dtype, device=self.device)
        
        # Compute a grid of all indices
        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='ij')

        # Reshape to [n_supercells, 3]
        supercell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=-1)
        
        # Calculate positions in real space
        # Result shape: [n_supercells, 3]
        supercell_positions = torch.matmul(supercell_indices, self.crystal.lattice_vectors)

        # Store original shape of q_vectors for later reshaping
        q_size_original = q_vectors.shape[:-1]
        
        # Reshape q_vectors to [n_pixels, 3] for matrix multiplication
        q_vectors_flat = q_vectors.view(-1, 3)
        n_pixels = q_vectors_flat.shape[0]

        # Calculate averaged displacements
        batch_size = displacement_field.shape[0]
        n_atoms = displacement_field.shape[-2]

        # Check displacement field dimensions and convert to supercell level if needed
        if displacement_field.shape[-5:-2] == (n1, n2, n3):
            # Displacement field is at unit cell resolution - need to downsample to supercell resolution
            # Reshape to group unit cells into supercells: [batch_size, n_sc1, d1, n_sc2, d2, n_sc3, d3, n_atoms, 3]
            grouped = displacement_field.view(batch_size, n_sc1, d1, n_sc2, d2, n_sc3, d3, n_atoms, 3)
            
            # Average over supercell dimensions (2, 4, 6) -> [n_sc1, n_sc2, n_sc3, n_atoms, 3]
            avg_displacements = torch.mean(grouped, dim=(2, 4, 6))
            
        elif displacement_field.shape[-5:-2] == (n_sc1, n_sc2, n_sc3):
            # Displacement field is already at supercell resolution
            avg_displacements = displacement_field
            
        else:
            raise ValueError(
                f"Displacement field shape {displacement_field.shape[-5:-2]} is incompatible with crystal grid "
                f"of size ({n1}, {n2}, {n3}) or supercell grid ({n_sc1}, {n_sc2}, {n_sc3})"
            )
        
        # Flatten to [batch_size, n_supercells, n_atoms, 3]
        avg_displacements_flat = avg_displacements.view(batch_size, -1, n_atoms, 3)

        # Determine batch size
        if q_batch_size is None:
            q_batch_size = n_pixels

        # Process in batches
        scattering_results = []
        
        for i in range(0, n_pixels, q_batch_size):
            q_batch = q_vectors_flat[i:i+q_batch_size]
            
            # Calculate q·r for each atom in the unit cell and each q-vector
            # q_vectors_flat shape: [n_pixels, 3]
            # atom_positions shape: [n_atoms, 3]
            # Result shape: [n_pixels, n_atoms]
            q_dot_r = torch.matmul(q_batch, self.crystal.atom_positions.T)
    
            # Calculate e^(-iq·r) for each atom and each q-vector
            basis_phase_factors = torch.exp(-1j * q_dot_r)
    
            # Calculate q·u_m for each supercell, each atom in the unit cell, and each q-vector
            # q_vectors_flat shape: [n_pixels, 3]
            # avg_displacements_flat shape: [batch_size, n_supercells, n_atoms, 3]
            # Result shape: [batch_size, n_pixels, n_supercells, n_atoms]
            q_dot_displacements = torch.einsum('pi,bnmi->bpnm', q_batch, avg_displacements_flat)
    
            # Calculate e^(-iq·u_m) for each supercell, each atom in the unit cell, and each q-vector
            displacement_phase_factors = torch.exp(-1j * q_dot_displacements)
    
            # Calculate total phase factors for all supercells
            # total phase = basis_phase * displacement_phase
            # basis_phase_factors shape: [n_pixels, n_atoms] -> [1, n_pixels, 1, n_atoms]  
            # displacement_phase_factors shape: [batch_size, n_pixels, n_supercells, n_atoms]
            # Result shape: [batch_size, n_pixels, n_supercells, n_atoms]
            total_phase_factors = basis_phase_factors.unsqueeze(1).unsqueeze(0) * displacement_phase_factors
    
            # Calculate |q| for form factor (convert to Å for typical form factor formulas)
            # Result shape: [n_pixels, 1]
            q_magnitude = torch.norm(q_batch, dim=-1, keepdim=True) * 1e-10
    
            # Vectorized calculation of form factors for all atoms and all q values
            form_factors = self.crystal.calculate_form_factors(q_magnitude)
            
            # Calculate modified structure factors for all supercells
            # form_factors shape: [n_pixels, n_atoms] -> [1, n_pixels, 1, n_atoms]
            # Result shape: [batch_size, n_pixels, n_supercells]
            modified_structure_factors = torch.sum(form_factors.unsqueeze(1).unsqueeze(0) * total_phase_factors, dim=-1)
    
            # Apply mask if provided (maintains differentiability)
            if mask is not None:
                # Check mask dimensions and convert to supercell level if needed
                if mask.shape[-3:] == (n1, n2, n3):
                    # Mask is at unit cell resolution - need to downsample to supercell resolution
                    # Reshape mask to group unit cells into supercells
                    mask_grouped = mask.view(-1, n_sc1, d1, n_sc2, d2, n_sc3, d3)
                    
                    # Average over supercell dimensions
                    supercell_mask = torch.mean(mask_grouped, dim=(2, 4, 6))
                    
                elif mask.shape[-3:] == (n_sc1, n_sc2, n_sc3):
                    # Mask is already at supercell resolution
                    supercell_mask = mask
                    
                else:
                    raise ValueError(
                        f"Mask shape {mask.shape[-3:]} is incompatible with crystal grid "
                        f"of size ({n1}, {n2}, {n3}) or supercell grid ({n_sc1}, {n_sc2}, {n_sc3})"
                    )
                
                # Flatten supercell mask to [batch_size, n_supercells]
                supercell_mask_flat = supercell_mask.flatten(start_dim=1)
                
                # Apply mask to modified structure factors via element-wise multiplication
                # modified_structure_factors shape: [batch_size, n_pixels, n_supercells]
                # supercell_mask_flat shape: [batch_size, n_supercells] -> [batch_size, 1, n_supercells]
                modified_structure_factors = modified_structure_factors * supercell_mask_flat.unsqueeze(1)
    
            # Calculate q·R for each supercell and each q-vector
            # Result shape: [n_pixels, n_supercells]
            q_dot_R = torch.matmul(q_batch, supercell_positions.T)
    
            # Calculate e^(-iq·R) for each supercell and each q-vector
            supercell_phase_factors = torch.exp(-1j * q_dot_R)
    
            # Number of unit cells per supercell
            cells_per_supercell = d1 * d2 * d3
            
            # Multiply modified structure factor by sum of phase factors and scale by supercell size
            # modified_structure_factors shape: [batch_size, n_pixels, n_supercells]
            # supercell_phase_factors shape: [n_pixels, n_supercells]
            # Result shape: [batch_size, n_pixels]
            scattering_batch = torch.sum(modified_structure_factors * supercell_phase_factors.unsqueeze(0), dim=-1) * cells_per_supercell
    
            # Add global position phase shift
            # Calculate q·R_g for each q-vector
            # grain_position shape: [3]
            # q_vectors_flat shape: [n_pixels, 3]
            # Result shape: [n_pixels]
            q_dot_global_position = torch.matmul(q_batch, self.crystal.position)
            
            # Calculate e^(-iq·R_g) for each q-vector
            global_phase_factor = torch.exp(-1j * q_dot_global_position)
            
            # Apply global position phase shift
            scattering_batch = scattering_batch * global_phase_factor.unsqueeze(0)

            # Append scattering results
            scattering_results.append(scattering_batch)

        # Concatenate along q-vector dimension
        scattering_amplitude = torch.cat(scattering_results, dim=1)
    
        # Reshape back to original q_vector shape
        scattering_amplitude = scattering_amplitude.view((batch_size,) + q_size_original)
        
        return scattering_amplitude