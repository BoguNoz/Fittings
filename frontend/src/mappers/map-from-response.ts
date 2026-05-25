import type {PTRChartData, PTRChartPoint} from "../models/data-responses.ts";

export const mapPtrResponse = (json: any): PTRChartData => {
    return {
        norm_amplitude_data: json.norm_amplitude_data?.map((series: any[]) =>
            series.map((point: any) => [Number(point[0]), Number(point[1])] as PTRChartPoint)
        ) ?? [],

        phase_data: json.phase_data?.map((series: any[]) =>
            series.map((point: any) => [Number(point[0]), Number(point[1])] as PTRChartPoint)
        ) ?? [],

        results: {
            anisotropy: Number(json.results?.anisotropy ?? 0),
            k2: Number(json.results?.k2 ?? 0),
            alfa2: Number(json.results?.alfa2r ?? 0),
            r32: Number(json.results?.r32 ?? 0),
            kParallel: Number(json.results?.kParallel ?? 0),
            r2Amp: Number(json.results?.r2Amp ?? 0),
            r2Phase: Number(json.results?.r2Phase ?? 0),
            resNorm: Number(json.results?.resNorm ?? 0),
        },
    };
};