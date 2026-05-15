export interface PTRChartPoint {
    0: number;  // x (frequency or log10 frequency)
    1: number;  // y (amplitude or phase)
}

export interface PTRResults {
    sample_name: string;
    k2: number;
    alfa2: number;
    r32: number;
    phi0_deg: number;
    res_norm: number;
}

export interface PTRChartData {
    amplitude_data: PTRChartPoint[][];        // [ [Model series], [Experiment series] ]
    norm_amplitude_data: PTRChartPoint[][];   // Normalized log-log amplitude
    phase_data: PTRChartPoint[][];            // Phase data
    results: PTRResults;
}