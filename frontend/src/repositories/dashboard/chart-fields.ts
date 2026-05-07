import {buildFields, createFieldPlaceholders} from "@bogunoz/simplify";
import {lang} from "../../text/utils/lang.ts";
import {formStore} from "../../stores/form-store.ts";


// #region PTR Amplitude
const text = lang();
const al = createFieldPlaceholders({amplitudeLinear: "amplitudeLinear"}, text.ptrPlots);

al.amplitudeLinear.render = true
al.amplitudeLinear.dataSource = () => {
    return formStore.ptrData?.amplitude_data ?? [];
};
al.amplitudeLinear.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField("amplitudeLinear", callback);
};

export const amplitudeLinearField = buildFields(al);
// #endregion PTR Amplitude

// #region Amplitude
const nal = createFieldPlaceholders({normalizedAmplitudeLog: "normalizedAmplitudeLog"}, text.ptrPlots);

nal.normalizedAmplitudeLog.render = true
nal.normalizedAmplitudeLog.dataSource = () => {
    return formStore.ptrData?.norm_amplitude_data ?? [];
};
nal.normalizedAmplitudeLog.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField("normalizedAmplitudeLog", callback);
};

export const normalizedAmplitudeLogField = buildFields(nal);
// #endregion Amplitude

// #region PTR Phase
const p = createFieldPlaceholders({phase: "phase"}, text.ptrPlots);


p.phase.render = true
p.phase.dataSource = () => {
    return formStore.ptrData?.phase_data ?? [];
};
p.phase.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField("phase", callback);
};

export const phaseField = buildFields(p);
// #endregion PTR Phase