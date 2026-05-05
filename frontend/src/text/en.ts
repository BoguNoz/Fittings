
export default {

    validators: {
        validateThickness: "Value seems unusually large. Did you provide it in meters (e.g., 240e-9)?",
        validateDiffusivity: "Unusual order of magnitude detected for diffusivity. Please check the units [m²/s].",
    },

    ptrConfig: {
        sectionTitle: "PTR Configuration",
        sectionDescription: "Configuration parameters for the PTR model fitting. Contains fixed geometrical and thermal properties of the layers and interfacial resistances.",

        // Layer 2 – Main ZnO layer
        l2Label: "ZnO Layer Thickness (L2)",
        l2Description: "Thickness of the main ZnO layer (Layer 2), expressed in meters. Usually determined from RBS or ellipsometry measurements.",

        // Layer 1 – Top / buffer layer
        k1Label: "Thermal Conductivity Layer 1 (K1)",
        k1Description: "Thermal conductivity of the top layer (Layer 1) in W/(m·K).",

        l1Label: "Thickness of Layer 1 (L1)",
        l1Description: "Thickness of the top layer (Layer 1), expressed in meters.",

        alfa1Label: "Thermal Diffusivity Layer 1 (α1)",
        alfa1Description: "Thermal diffusivity of Layer 1 in m²/s.",

        // Layer 3 – Substrate
        alfa3Label: "Thermal Diffusivity Substrate (α3)",
        alfa3Description: "Thermal diffusivity of the substrate (Layer 3) in m²/s.",

        // Interfacial resistance
        r21Label: "Thermal Boundary Resistance L2–L1 (R21)",
        r21Description: "Interfacial thermal resistance between ZnO layer (Layer 2) and top layer (Layer 1) in m²·K/W.",
    },

    ptrFitResult: {
        sectionTitle: "PTR Fitting Results",
        sectionDescription: "Container holding the results of the photothermal radiometry (PTR) model fitting, including optimized thermal parameters, model curves, and fit diagnostics.",

        k2Label: "Thermal Conductivity (ZnO Layer)",
        k2Description: "Thermal conductivity of the ZnO thin film (Layer 2) in W/(m·K).",

        alfa2Label: "Thermal Diffusivity (ZnO Layer)",
        alfa2Description: "Thermal diffusivity of the ZnO thin film (Layer 2) in m²/s.",

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
    }
}