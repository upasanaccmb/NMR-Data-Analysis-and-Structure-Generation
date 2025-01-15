import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import nmrglue as ng
import pandas as pd
from scipy import signal, optimize
from dataclasses import dataclass
from pathlib import Path
from Bio.PDB import *
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import logging
from concurrent.futures import ThreadPoolExecutor
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NMRProcessingError(Exception):
    """Custom exception for NMR processing errors"""
    pass

@dataclass
class NMRSpectrum:
    """Enhanced class to hold processed NMR spectrum data"""
    dic: dict
    data: np.ndarray
    peaks: List[Tuple[float, ...]]
    assignments: Dict[str, Tuple[float, ...]]
    spectrum_type: str  # e.g., 'HSQC', 'NOESY', 'TOCSY'
    processed: bool = False
    snr: float = 0.0  # Signal-to-noise ratio
    linewidths: Dict[str, float] = None  # Peak linewidths

    def validate(self):
        """Validate spectrum data and parameters"""
        if self.data is None or len(self.data.shape) != 2:
            raise NMRProcessingError("Invalid spectrum data format")
        if not self.dic:
            raise NMRProcessingError("Missing spectrum parameters")
        return True

class StructureQuality:
    """Class to hold structure quality metrics"""
    def __init__(self):
        self.ramachandran_scores = {}
        self.clash_score = 0.0
        self.noe_violations = []
        self.secondary_structure = {}
        self.rmsd = 0.0

class NMRPipeline:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create directories for intermediate results
        self.processed_dir = self.output_dir / "processed_spectra"
        self.assignments_dir = self.output_dir / "assignments"
        self.structures_dir = self.output_dir / "structures"
        self.quality_dir = self.output_dir / "quality_analysis"
        
        for directory in [self.processed_dir, self.assignments_dir, 
                         self.structures_dir, self.quality_dir]:
            directory.mkdir(exist_ok=True)

        # Initialize processing parameters
        self.params = self._load_default_parameters()
        
        # Setup logging
        self._setup_logging()

    def _load_default_parameters(self) -> dict:
        """Load default processing parameters"""
        return {
            'peak_picking': {
                'threshold': 0.1,
                'noise_threshold': 0.05,
                'min_peak_distance': 0.1,
                'max_peaks': 10000
            },
            'phase_correction': {
                'ph0_max': 360,
                'ph1_max': 360,
                'optimization_method': 'Nelder-Mead'
            },
            'structure_calculation': {
                'noe_upper_distance': 6.0,
                'noe_lower_distance': 1.8,
                'sa_temp_high': 1000,
                'sa_temp_low': 100,
                'sa_steps': 50000
            }
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def load_spectrum(self, spectrum_path: str, spectrum_type: str) -> NMRSpectrum:
        """Enhanced spectrum loading with validation and error handling"""
        try:
            logger.info(f"Loading {spectrum_type} spectrum from {spectrum_path}")
            
            # Load raw data
            dic, data = ng.bruker.read(spectrum_path)
            
            # Validate data dimensions
            if len(data.shape) != 2:
                raise NMRProcessingError(f"Expected 2D data, got {len(data.shape)}D")
                
            # Create spectrum object
            spectrum = NMRSpectrum(
                dic=dic,
                data=data,
                peaks=[],
                assignments={},
                spectrum_type=spectrum_type,
                linewidths={}
            )
            
            # Validate spectrum
            spectrum.validate()
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Error loading spectrum: {str(e)}")
            raise NMRProcessingError(f"Failed to load spectrum: {str(e)}")

    def _optimize_phase_correction(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Optimize phase correction parameters using entropy minimization
        
        Returns:
            Tuple of optimized zero- and first-order phase corrections
        """
        def entropy(params):
            ph0, ph1 = params
            phased_data = self._apply_phase_correction(data, ph0, ph1)
            # Calculate entropy of real part
            prob = np.abs(phased_data.real) / np.sum(np.abs(phased_data.real))
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            return entropy

        # Optimize using Nelder-Mead
        result = optimize.minimize(
            entropy,
            x0=[0, 0],
            method='Nelder-Mead',
            bounds=[(-360, 360), (-360, 360)]
        )
        
        return result.x[0], result.x[1]

    def _apply_phase_correction(self, data: np.ndarray, ph0: float, ph1: float) -> np.ndarray:
        """Apply phase correction with given parameters"""
        ph0_rad = np.deg2rad(ph0)
        ph1_rad = np.deg2rad(ph1)
        
        # Create phase correction array
        x = np.linspace(0, 1, data.shape[1])
        phase = ph0_rad + ph1_rad * x
        
        # Apply phase correction
        phased_data = data * np.exp(1j * phase)
        
        return phased_data

    def process_spectrum(self, spectrum: NMRSpectrum) -> NMRSpectrum:
        """
        Enhanced spectrum processing with sophisticated algorithms
        """
        logger.info(f"Processing {spectrum.spectrum_type} spectrum")
        
        try:
            # Apply window function
            data = self._apply_window_function(spectrum.data)
            
            # Optimize and apply phase correction
            ph0, ph1 = self._optimize_phase_correction(data)
            data = self._apply_phase_correction(data, ph0, ph1)
            
            # Perform Fourier transform
            data = np.fft.fft2(data)
            
            # Baseline correction
            data = self._baseline_correction(data)
            
            # Calculate SNR
            spectrum.snr = self._calculate_snr(data)
            
            # Update spectrum object
            spectrum.data = data
            spectrum.processed = True
            
            # Save processed spectrum
            self._save_processed_spectrum(spectrum)
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Error processing spectrum: {str(e)}")
            raise NMRProcessingError(f"Failed to process spectrum: {str(e)}")

    def _baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """
        Perform automated baseline correction using assymetric least squares
        """
        def alsbaseline(y, lam=1e5, p=0.01, niter=10):
            L = len(y)
            D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.diags(w, 0, shape=(L,L))
                Z = W + lam * D.dot(D.transpose())
                z = sparse.linalg.spsolve(Z, w*y)
                w = p * (y > z) + (1-p) * (y < z)
            return z

        # Apply to each row
        corrected = np.zeros_like(data)
        for i in range(data.shape[0]):
            corrected[i] = data[i] - alsbaseline(np.real(data[i]))
            
        return corrected

    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Estimate noise from corner of spectrum
        noise_region = data[:50, :50]
        noise = np.std(np.abs(noise_region))
        
        # Calculate signal as mean of top peaks
        signal = np.mean(np.sort(np.abs(data.flatten()))[-100:])
        
        return signal/noise

    def pick_peaks(self, spectrum: NMRSpectrum) -> List[Tuple[float, ...]]:
        """
        Enhanced peak picking with validation and clustering
        """
        logger.info(f"Picking peaks for {spectrum.spectrum_type} spectrum")
        
        try:
            # Find local maxima
            peaks = signal.find_peaks_2d(
                np.abs(spectrum.data),
                threshold=self.params['peak_picking']['threshold'],
                distance=self.params['peak_picking']['min_peak_distance']
            )
            
            # Convert to array for clustering
            peak_array = np.array(peaks)
            
            # Cluster peaks to remove artifacts
            clusterer = DBSCAN(eps=3, min_samples=2)
            clusters = clusterer.fit_predict(peak_array)
            
            # Filter noise peaks
            valid_peaks = peak_array[clusters != -1]
            
            # Convert to ppm values
            ppm_peaks = [self._index_to_ppm(peak, spectrum.dic) for peak in valid_peaks]
            
            # Calculate linewidths
            spectrum.linewidths = self._calculate_linewidths(spectrum.data, valid_peaks)
            
            # Update spectrum
            spectrum.peaks = ppm_peaks
            
            # Save peaks
            self._save_peaks(spectrum)
            
            return ppm_peaks
            
        except Exception as e:
            logger.error(f"Error picking peaks: {str(e)}")
            raise NMRProcessingError(f"Failed to pick peaks: {str(e)}")

    def _calculate_linewidths(self, data: np.ndarray, peak_positions: np.ndarray) -> Dict[str, float]:
        """Calculate linewidths for each peak"""
        linewidths = {}
        
        for i, peak in enumerate(peak_positions):
            # Extract region around peak
            region = self._extract_peak_region(data, peak)
            
            # Fit Lorentzian function
            try:
                width = self._fit_lorentzian(region)
                linewidths[f"peak_{i}"] = width
            except:
                logger.warning(f"Failed to fit linewidth for peak {i}")
                
        return linewidths

    def assign_backbone(self, hsqc: NMRSpectrum, sequence: str) -> Dict[str, Tuple[float, ...]]:
        """
        Enhanced backbone assignment with validation
        """
        logger.info("Starting backbone assignment")
        
        try:
            # Load chemical shift statistics
            cs_stats = self._load_chemical_shift_statistics()
            
            # Initialize assignments
            assignments = {}
            
            # For each amino acid type, create probability distribution
            type_probabilities = self._calculate_type_probabilities(hsqc.peaks, cs_stats)
            
            # Sequential assignment using optimization
            assignments = self._optimize_sequential_assignment(
                peaks=hsqc.peaks,
                type_probs=type_probabilities,
                sequence=sequence
            )
            
            # Validate assignments
            self._validate_assignments(assignments, sequence)
            
            # Save assignments
            self._save_assignments(assignments, "backbone")
            
            return assignments
            
        except Exception as e:
            logger.error(f"Error in backbone assignment: {str(e)}")
            raise NMRProcessingError(f"Failed to assign backbone: {str(e)}")

    def assign_noes(self, 
                   noesy: NMRSpectrum, 
                   assignments: Dict[str, Tuple[float, ...]]
                   ) -> List[Tuple[str, str, float]]:
        """
        Enhanced NOE assignment with sophisticated distance calculation
        """
        logger.info("Starting NOE assignment")
        
        try:
            noe_restraints = []
            
            # Create chemical shift lookup
            shift_lookup = self._create_shift_lookup(assignments)
            
            # For each NOESY peak
            for peak in noesy.peaks:
                # Match chemical shifts
                matches = self._find_shift_matches(peak, shift_lookup)
                
                if matches:
                    # Calculate distance from volume
                    volume = self._calculate_peak_volume(noesy.data, peak)
                    distance = self._volume_to_distance(volume)
                    
                    # Add restraint
                    noe_restraints.append((
                        matches[0],
                        matches[1],
                        distance
                    ))
            
            # Apply symmetry
            noe_restraints = self._symmetrize_restraints(noe_restraints)
            
            # Remove duplicates and validate
            noe_restraints = self._validate_restraints(noe_restraints)
            
            # Save restraints
            self._save_restraints(noe_restraints)
            
            return noe_restraints
            
        except Exception as e:
            logger.error(f"Error in NOE assignment: {str(e)}")
            raise NMRProcessingError(f"Failed to assign NOEs: {str(e)}")

    def calculate_structure(self,
                          sequence: str,
                          noe_restraints: List[Tuple[str, str, float]],
                          output_pdb: str):
        """
        Enhanced structure calculation with validation
        """
        logger.info("Starting structure calculation")
        
        try:
            # Initialize CNS or CYANA interface
            calculator = self._initialize_structure_calculator()
            
            # Generate extended chain
            initial_structure = self._generate_extended_chain(sequence)
            
            # Setup force field
            self._setup_force_field(calculator)
            
            # Add restraints
            self._add_restraints(calculator, noe_restraints)
            
            # Run simulated annealing
            structures = self._run_simulated_annealing(calculator)
            
            # Select best structures
            best_structures = self._select_best_structures(structures)
            
            # Minimize final structures
            final_structures = self._minimize_structures(best_structures)
            
            # Analyze quality
            quality = self._analyze_structure_quality(final_structures)
            
            # Save structures and quality report
            self._save_structures(final_structures, output_pdb)
            self._save_quality_report(quality)
            
        except Exception as e:
            logger.error(f"Error in structure calculation: {str(e)}")
            raise


def _analyze_structure_quality(self, structures: List[Structure]) -> StructureQuality:
    """
    Analyze quality of calculated structures
    """
    quality = StructureQuality()
    
    # Calculate RMSD
    quality.rmsd = self._calculate_rmsd(structures)
    
    # Analyze Ramachandran plot regions
    quality.ramachandran_scores = self._analyze_ramachandran(structures)
    
    # Check for steric clashes
    quality.clash_score = self._calculate_clash_score(structures)
    
    # Find NOE violations
    quality.noe_violations = self._find_noe_violations(structures)
    
    # Determine secondary structure
    quality.secondary_structure = self._analyze_secondary_structure(structures)
    
    return quality

def _calculate_rmsd(self, structures: List[Structure]) -> float:
    """Calculate RMSD between structures"""
    # Use Bio.PDB's Superimposer
    super_imposer = Superimposer()
    reference = structures[0]
    
    rmsd_sum = 0
    for structure in structures[1:]:
        # Superimpose structures
        super_imposer.set_atoms(reference.get_atoms(), structure.get_atoms())
        rmsd_sum += super_imposer.rms
        
    return rmsd_sum / (len(structures) - 1)

def _analyze_ramachandran(self, structures: List[Structure]) -> Dict[str, float]:
    """Analyze Ramachandran plot regions for structures"""
    scores = {
        'favored': 0,
        'allowed': 0,
        'outlier': 0
    }
    
    for structure in structures:
        # Calculate phi/psi angles
        phi_psi = self._calculate_phi_psi(structure)
        
        # Check each residue against Ramachandran regions
        for residue, (phi, psi) in phi_psi.items():
            region = self._determine_ramachandran_region(phi, psi)
            scores[region] += 1
            
    # Convert to percentages
    total = sum(scores.values())
    return {k: (v/total)*100 for k, v in scores.items()}

def _calculate_clash_score(self, structures: List[Structure]) -> float:
    """Calculate clash score based on van der Waals violations"""
    clash_count = 0
    total_atoms = 0
    
    for structure in structures:
        # Get atom pairs within cutoff
        ns = NeighborSearch(list(structure.get_atoms()))
        
        # Check for clashes
        for atom1 in structure.get_atoms():
            close_atoms = ns.search(atom1.coord, 2.0)  # 2.0 Å cutoff
            for atom2 in close_atoms:
                if atom1 != atom2:
                    # Calculate VDW violation
                    distance = atom1 - atom2
                    vdw_sum = self._get_vdw_radius(atom1) + self._get_vdw_radius(atom2)
                    if distance < (vdw_sum * 0.75):  # 0.75 factor for severe clash
                        clash_count += 1
                        
        total_atoms += len(list(structure.get_atoms()))
        
    return (clash_count / total_atoms) * 1000  # Clashes per 1000 atoms

def _find_noe_violations(self, structures: List[Structure]) -> List[Dict]:
    """Find NOE distance restraint violations"""
    violations = []
    
    for structure in structures:
        structure_violations = []
        
        # Check each NOE restraint
        for restraint in self.noe_restraints:
            atom1, atom2, target_distance = restraint
            
            # Get actual distance in structure
            actual_distance = self._calculate_atom_distance(structure, atom1, atom2)
            
            # Check for violation
            violation = actual_distance - target_distance
            if abs(violation) > 0.5:  # 0.5 Å threshold
                structure_violations.append({
                    'atoms': (atom1, atom2),
                    'target': target_distance,
                    'actual': actual_distance,
                    'violation': violation
                })
                
        violations.append(structure_violations)
        
    return violations

def _analyze_secondary_structure(self, structures: List[Structure]) -> Dict[str, Dict[str, float]]:
    """Analyze secondary structure elements"""
    ss_analysis = {}
    
    for i, structure in enumerate(structures):
        # Use DSSP to calculate secondary structure
        dssp = DSSP(structure[0], 'temp.pdb')  # Temporary PDB file
        
        structure_ss = {
            'helix': 0,
            'sheet': 0,
            'coil': 0
        }
        
        # Count secondary structure elements
        for residue in dssp:
            ss = residue[2]  # DSSP code
            if ss in ['H', 'G', 'I']:  # Helical
                structure_ss['helix'] += 1
            elif ss in ['B', 'E']:  # Sheet
                structure_ss['sheet'] += 1
            else:  # Coil
                structure_ss['coil'] += 1
                
        ss_analysis[f'model_{i+1}'] = structure_ss
        
    return ss_analysis

def _generate_quality_plots(self, quality: StructureQuality):
    """Generate quality analysis plots"""
    # Create output directory for plots
    plot_dir = self.quality_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    # Ramachandran plot
    self._plot_ramachandran(quality.ramachandran_scores, plot_dir)
    
    # NOE violations plot
    self._plot_noe_violations(quality.noe_violations, plot_dir)
    
    # Secondary structure plot
    self._plot_secondary_structure(quality.secondary_structure, plot_dir)

def _plot_ramachandran(self, scores: Dict[str, float], plot_dir: Path):
    """Create Ramachandran plot with statistics"""
    plt.figure(figsize=(8, 6))
    
    # Create pie chart of Ramachandran statistics
    plt.pie(scores.values(), labels=scores.keys(), autopct='%1.1f%%')
    plt.title('Ramachandran Plot Statistics')
    
    plt.savefig(plot_dir / 'ramachandran_stats.png')
    plt.close()

def _plot_noe_violations(self, violations: List[Dict], plot_dir: Path):
    """Plot NOE violations distribution"""
    plt.figure(figsize=(10, 6))
    
    # Collect all violations
    all_violations = [v['violation'] for struct_violations in violations 
                     for v in struct_violations]
    
    # Create histogram
    plt.hist(all_violations, bins=20)
    plt.xlabel('Violation Distance (Å)')
    plt.ylabel('Count')
    plt.title('NOE Violations Distribution')
    
    plt.savefig(plot_dir / 'noe_violations.png')
    plt.close()

def _save_quality_report(self, quality: StructureQuality):
    """Generate and save quality report"""
    report = {
        'rmsd': quality.rmsd,
        'ramachandran_statistics': quality.ramachandran_scores,
        'clash_score': quality.clash_score,
        'noe_violations_summary': {
            'total': sum(len(v) for v in quality.noe_violations),
            'average_violation': np.mean([v['violation'] for struct_violations in quality.noe_violations 
                                       for v in struct_violations])
        },
        'secondary_structure': quality.secondary_structure
    }
    
    # Save as JSON
    with open(self.quality_dir / 'quality_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate plots
    self._generate_quality_plots(quality)

def run_pipeline(self, sequence: str):
    """Run complete pipeline with error handling and logging"""
    try:
        logger.info("Starting NMR structure determination pipeline")
        
        # Load and process spectra
        self.hsqc = self.load_spectrum(self.data_dir / "hsqc.ft2", "HSQC")
        self.noesy = self.load_spectrum(self.data_dir / "noesy.ft2", "NOESY")
        
        self.hsqc = self.process_spectrum(self.hsqc)
        self.noesy = self.process_spectrum(self.noesy)
        
        # Peak picking
        self.pick_peaks(self.hsqc)
        self.pick_peaks(self.noesy)
        
        # Assignments
        assignments = self.assign_backbone(self.hsqc, sequence)
        noe_restraints = self.assign_noes(self.noesy, assignments)
        
        # Structure calculation
        output_pdb = self.structures_dir / "final_structure.pdb"
        self.calculate_structure(
            sequence=sequence,
            noe_restraints=noe_restraints,
            output_pdb=output_pdb
        )
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage with error handling
    try:
        pipeline = NMRPipeline(
            data_dir="raw_data",
            output_dir="processed_data"
        )
        
        sequence = "MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVL"
        pipeline.run_pipeline(sequence)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise

