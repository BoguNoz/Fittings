
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
    },

    ptrConfig: {
        sectionTitle: "PTR Configuration",
        sectionDescription: "Configuration parameters for the PTR model fitting. Contains fixed geometrical and thermal properties of the layers and interfacial resistances.",

        l2Label: "ZnO Layer Thickness (L2)",
        l2Description: "Thickness of the main ZnO layer (Layer 2), expressed in meters. Usually determined from RBS or ellipsometry measurements.",

        k1Label: "Thermal Conductivity Layer 1 (K1)",
        k1Description: "Thermal conductivity of the top layer (Layer 1) in W/(m·K).",

        l1Label: "Thickness of Layer 1 (L1)",
        l1Description: "Thickness of the top layer (Layer 1), expressed in meters.",

        alfa1Label: "Thermal Diffusivity Layer 1 (α1)",
        alfa1Description: "Thermal diffusivity of the top layer (Layer 1) in m²/s.",

        // Poprawione:
        alfa2Label: "Thermal Diffusivity ZnO Layer (α2)",
        alfa2Description: "Thermal diffusivity of the ZnO thin film (Layer 2) in m²/s.",

        alfa3Label: "Thermal Diffusivity Substrate (α3)",
        alfa3Description: "Thermal diffusivity of the substrate (Layer 3) in m²/s.",

        r21Label: "Thermal Boundary Resistance L1–L2 (R21)",
        r21Description: "Interfacial thermal resistance between the top layer (Layer 1) and ZnO layer (Layer 2) in m²·K/W.",
    },

    ptrFitResult: {
        sectionTitle: "PTR Fitting Results",
        sectionDescription: "Container holding the results of the photothermal radiometry (PTR) model fitting, including optimized thermal parameters, model curves, and fit diagnostics.",

        k2Label: "Thermal Conductivity (ZnO Layer)",
        k2Description: "Thermal conductivity of the ZnO thin film (Layer 2) in W/(m·K).",

        alfa2rLabel: "Thermal Diffusivity (ZnO Layer)",
        alfa2rDescription: "Thermal diffusivity of the ZnO thin film (Layer 2) in m²/s.",

        r32Label: "Thermal Boundary Resistance (ZnO–Substrate)",
        r32Description: "Interfacial thermal resistance between the ZnO layer and the substrate (Layer 3) in m²·K/W.",

        k3Label: "Thermal Conductivity (Substrate)",
        k3Description: "Thermal conductivity of the substrate (Layer 3) in W/(m·K).",

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

        l2Label: "ZnO Layer Thickness",
        l2Description: "Thickness of the ZnO layer (Layer 2) used in the model, in meters.",

        sampleNameLabel: "Sample Name",
        sampleNameDescription: "Identifier of the analyzed sample."
    },

    errors: {
        fetchPtrResultsFailed: "Failed to fetch data.",
    }
}