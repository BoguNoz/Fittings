export interface PTRChartPoint {
    0: number;  // x (frequency or log10 frequency)
    1: number;  // y (amplitude or phase)
}

export interface PTRResults {
    anisotropy: number,
    k2: number,
    alfa2: number,
    r32: number,
    kParallel: number,
    r2Amp: number,
    r2Phase: number,
    resNorm: number,
}

export interface PTRChartData {
    norm_amplitude_data: PTRChartPoint[][];   // Normalized log-log amplitude
    phase_data: PTRChartPoint[][];            // Phase data
    results: PTRResults;
}