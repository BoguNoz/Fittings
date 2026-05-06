import {buildFields, createFieldPlaceholders} from "@bogunoz/simplify";
import {lang} from "../../text/utils/lang.ts";

const generateMultiSineData = (points = 200): [number, number][][] => {
    const sin: [number, number][] = [];
    const cos: [number, number][] = [];

    for (let i = 0; i < points; i++) {
        const x = i * 0.1;
        sin.push([x, Math.sin(x)]);
        cos.push([x, Math.cos(x)]);
    }

    return [sin, cos];
};

// #region PTR Amplitude
const text = lang();
const al = createFieldPlaceholders({amplitudeLinear: "amplitudeLinear"}, text.ptrPlots);

al.amplitudeLinear.render = true
al.amplitudeLinear.dataSource = () => {
    return generateMultiSineData();
}

export const amplitudeLinearField = buildFields(al);
// #endregion PTR Amplitude

// #region Amplitude
const nal = createFieldPlaceholders({normalizedAmplitudeLog: "normalizedAmplitudeLog"}, text.ptrPlots);

nal.normalizedAmplitudeLog.render = true
nal.normalizedAmplitudeLog.dataSource = () => {
    return generateMultiSineData();
}

export const normalizedAmplitudeLogField = buildFields(nal);
// #endregion Amplitude

// #region PTR Phase
const p = createFieldPlaceholders({phase: "phase"}, text.ptrPlots);

p.phase.render = true
p.phase.dataSource = () => {
    return generateMultiSineData();
}

export const phaseField = buildFields(p);
// #endregion PTR Phase