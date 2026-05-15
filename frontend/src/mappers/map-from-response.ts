import type {PTRChartData, PTRChartPoint} from "../models/data-responses.ts";

export const mapPtrResponse = (json: any): PTRChartData => {
    return {
        amplitude_data: json.amplitude_data?.map((series: any[]) =>
            series.map((point: any) => [Number(point[0]), Number(point[1])] as PTRChartPoint)
        ) ?? [],

        norm_amplitude_data: json.norm_amplitude_data?.map((series: any[]) =>
            series.map((point: any) => [Number(point[0]), Number(point[1])] as PTRChartPoint)
        ) ?? [],

        phase_data: json.phase_data?.map((series: any[]) =>
            series.map((point: any) => [Number(point[0]), Number(point[1])] as PTRChartPoint)
        ) ?? [],

        results: {
            sample_name: String(json.results?.sample_name ?? ""),
            k2: Number(json.results?.k2 ?? 0),
            alfa2: Number(json.results?.alfa2 ?? 0),
            r32: Number(json.results?.r32 ?? 0),
            phi0_deg: Number(json.results?.phi0_deg ?? 0),
            res_norm: Number(json.results?.res_norm ?? 0),
        },
    };
};