
export default {

    validators: {
        validateThickness: "Value seems unusually large. Did you provide it in meters (e.g., 240e-9)?",
        validateDiffusivity: "Unusual order of magnitude detected for diffusivity. Please check the units [m²/s].",
    },

    actions: {
        generateButtonLabel: "Generate",
    },

    ptrExecution: {
        sectionTitle: "Experimental Data & Fitting Execution",
        sectionDescription: "Parameters for experimental data ingestion, sample identification, and numerical optimization of the photothermal response.",
    },

    ptrPlots: {
        sectionTitle: "PTR Fitting Plots",
        sectionDescription: "Visualizations of the experimental data versus the fitted theoretical model.",

        // Plot 1
        amplitudeLinearLabel: "PTR Amplitude (Linear Scale)",
        amplitudeLinearDescription: "PTR signal amplitude as a function of modulation frequency shown on a linear amplitude scale. Compares experimental data with the best-fit model.",

        // Plot 2
        normalizedAmplitudeLogLabel: "Normalized Amplitude (Log-Log Scale)",
        normalizedAmplitudeLogDescription: "Log-log plot of normalized PTR amplitude versus frequency. Highlights the frequency-dependent decay of the thermal wave and allows easy visual assessment of the fit quality.",

        // Plot 3
        phaseLabel: "PTR Phase Response",
        phaseDescription: "Phase of the PTR signal as a function of modulation frequency (log scale). Most sensitive to thermal diffusivity and thermal boundary resistance.",

        // Common labels used in plots
        frequencyLabel: "Frequency [Hz]",
        logFrequencyLabel: "log₁₀(Frequency) [Hz]",
        modelLabel: "Model",
        experimentLabel: "Experiment",
        amplitudeLabel: "Amplitude",
        normalizedAmplitudeLabel: "log₁₀(Normalized Amplitude)",
        phaseDegLabel: "Phase [deg]"
    },

    ptr: {
        sectionTitle: "PTR Forward Models",
        sectionDescription: "Available simulation models for photothermal radiometry signal calculation.",

        dataInputLabel: "Load .dat File",
        dataInputDescription: "Loads experimental PTR data from a .dat file (frequency, amplitude, phase).",

        sampleNameLabel: "Sample Name",
        sampleNameDescription: "Identifier of the sample (e.g., 'X32B', 'CT02'). Used for labeling plots and results.",

        useFullHankelModelLabel: "Use Full Hankel Transform Model",
        useFullHankelModelDescription: "If enabled, uses the full 3D radial heat flow model (Hankel transform) which accounts for Gaussian beam profile and in-plane heat diffusion. If disabled, uses the simplified 1D thermal wave model (faster, but less accurate for beam size effects).",

        nStartsLabel: "Number of Optimization Starts",
        nStartsDescription: "Defines the number of independent initial points for the global optimization algorithm. Increasing this value enhances the probability of finding the global minimum in a complex, multi-dimensional parameter space.",

    },

    ptrConfig: {
        sectionTitle: "PTR Configuration",
        sectionDescription: "Fixed geometrical, thermal, and fitting parameters for the photothermal radiometry model.",

        l2Label: "Layer Thickness (L2)",
        l2Description: "Thickness of the film (Layer 2), in meters (e.g., 240e-9 or 480e-9).",

        k1Label: "Thermal Conductivity Layer 1 (K1)",
        k1Description: "Thermal conductivity of the top transducer/absorber layer (if present) in W/(m·K).",

        l1Label: "Thickness of Layer 1 (L1)",
        l1Description: "Thickness of the top layer (e.g., metal transducer), in meters.",

        alfa1Label: "Thermal Diffusivity Layer 1 (α1)",
        alfa1Description: "Thermal diffusivity of the top layer in m²/s.",

        alfa2Label: "Thermal Diffusivity (α2)",
        alfa2Description: "Thermal diffusivity of the layer (Layer 2) in m²/s.",

        alfa3Label: "Thermal Diffusivity Substrate (α3)",
        alfa3Description: "Thermal diffusivity of the substrate (e.g., glass) in m²/s.",

        r21Label: "Thermal Boundary Resistance L1–L2 (R21)",
        r21Description: "Interfacial thermal resistance between Layer 1 and layer in m²·K/W.",

        r32Label: "Thermal Boundary Resistance–Substrate (R32)",
        r32Description: "Interfacial thermal resistance between and the substrate in m²·K/W.",

        k2Label: "Thermal Conductivity (K2)",
        k2Description: "In-plane or cross-plane thermal conductivity of the layer in W/(m·K).",

        rhoc2Label: "Volumetric Heat Capacity (ρC2)",
        rhoc2Description: "Volumetric heat capacity of the film (Layer 2) in J/(m³·K).",

        k3Label: "Thermal Conductivity Substrate (K3)",
        k3Description: "Thermal conductivity of the substrate (e.g., glass) in W/(m·K).",

        dPumpLabel: "Pump Beam Diameter (d_pump)",
        dPumpDescription: "Diameter of the modulated laser beam (pump) at the sample surface, in meters (e.g., 2.4e-6).",

        qLabel: "Heating Power (Q)",
        qDescription: "Intensity or total power of the modulated pump beam, typically normalized or in Watts.",

        anisotropyLabel: "Thermal Anisotropy Factor",
        anisotropyDescription: "The ratio of in-plane to cross-plane thermal conductivity (K_in / K_cross) for the anisotropic layer.",

        weightLabel: "High-Frequency Weighting Exponent",
        weightDescription: "Power-law exponent used to weight residuals: weight = (f / f_max)^exponent. Increases importance of high-frequency points where thermal diffusion length is shorter. Controls balance between low- and high-frequency fit quality.",
    },

    ptrFitResult: {
        kParallelLabel: "Calculated In-plane Conductivity",
        kParallelDescription: "Derived thermal conductivity in the in-plane direction, calculated as k_cross × anisotropy.",

        r2AmpLabel: "Amplitude Fit Goodness (R²)",
        r2AmpDescription: "Coefficient of determination (R²) comparing the modeled amplitude to the experimental data.",

        r2PhaseLabel: "Phase Fit Goodness (R²)",
        r2PhaseDescription: "Coefficient of determination (R²) comparing the modeled phase to the experimental data.",

        anisotropyLabel: "Anisotropy Ratio",
        anisotropyDescription: "The ratio of in-plane to cross-plane thermal conductivity (K_parallel / K_cross).",

        k3Label: "Substrate Thermal Conductivity",
        k3Description: "Thermal conductivity of the substrate (Layer 3) used during the simulation, in W/(m·K).",

        sectionTitle: "PTR Fitting Results",
        sectionDescription: "Container holding the results of the photothermal radiometry (PTR) model fitting, including optimized thermal parameters, model curves, and fit diagnostics.",

        k2Label: "Thermal Conductivity",
        k2Description: "Fitted thermal conductivity of the thin film in W/(m·K). Literature values: ~0.3–0.5 W/(m·K) depending on thickness and direction.",

        alfa2rLabel: "Thermal Diffusivity",
        alfa2rDescription: "Fitted thermal diffusivity of the layer in m²/s. Literature range: ~1.2–1.8 × 10⁻⁷ m²/s (cross-plane).",

        r32Label: "Thermal Boundary Resistance",
        r32Description: "Interfacial thermal resistance between the second layer and the substrate (Layer 3) in m²·K/W.",

        phi0DegLabel: "Global Phase Offset",
        phi0DegDescription: "Global phase shift applied to align the model phase with experimental data, expressed in degrees.",

        resNormLabel: "Residual Norm",
        resNormDescription: "Optimization cost function value (2 × least-squares cost), representing the sum of squared weighted residuals.",

        modelAmpLabel: "Model Amplitude",
        modelAmpDescription: "Simulated amplitude response of the PTR model across the frequency vector.",

        modelPhaseDegLabel: "Model Phase (Degrees)",
        modelPhaseDegDescription: "Simulated phase response of the PTR model expressed in degrees.",

        expPhaseDegLabel: "Experimental Phase (Degrees)",
        expPhaseDegDescription: "Measured experimental phase data expressed in degrees.",

        phaseUnitsLabel: "Phase Units",
        phaseUnitsDescription: "Units used for the input phase data ('deg' or 'rad').",

        pfitLabel: "Optimized Parameters Vector",
        pfitDescription: "Vector of optimized parameters in log10 scale plus phi0.",

        exitFlagLabel: "Optimization Exit Flag",
        exitFlagDescription: "Status code returned by scipy.optimize.least_squares indicating the convergence result.",

        frequencyVectorLabel: "Frequency Vector",
        frequencyVectorDescription: "Array of modulation frequencies used in the PTR measurement.",

        l2Label: "Second Layer Thickness",
        l2Description: "Thickness of the second layer used in the model, in meters.",

        sampleNameLabel: "Sample Name",
        sampleNameDescription: "Identifier of the analyzed sample."
    },

    errors: {
        fetchPtrResultsFailed: "Failed to fetch data.",
    }
}