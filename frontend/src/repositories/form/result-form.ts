import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";
import {formStore} from "../../stores/form-store.ts";

export const resultFormRegisteredFields = {
    k2: "k2",
    alfa2r: "alfa2r",
    r32: "r32",
    k3: "k3",
    phi0Deg: "phi0Deg",
    resNorm: "resNorm",
}


const text = lang();
const fields = createFieldPlaceholders(resultFormRegisteredFields, text.ptrFitResult);

// #region K2
fields.k2.fieldType = BaseFieldTypesEnum.Input
fields.k2.isDisabled = true
fields.k2.variant = "secondary"
fields.k2.dataSource = () => {
    return formStore.ptrData?.results.k2 ?? "";
};
fields.k2.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField(resultFormRegisteredFields.k2, callback);
};

// #endregion K2

// #region K3
fields.k3.fieldType = BaseFieldTypesEnum.Input
fields.k3.isDisabled = true
fields.k3.variant = "secondary"
fields.k3.dataSource = () => {
    return formStore.ptrData?.results.k3 ?? "";
};
fields.k3.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField(resultFormRegisteredFields.k3, callback);
};
// #endregion K3

// #region Alfa2
fields.alfa2r.fieldType = BaseFieldTypesEnum.Input
fields.alfa2r.isDisabled = true
fields.alfa2r.variant = "secondary"
fields.alfa2r.dataSource = () => {
    return formStore.ptrData?.results.alfa2r ?? "";
};
fields.alfa2r.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField(resultFormRegisteredFields.alfa2r, callback);
};
// #endregion Alfa2

// #region R32
fields.r32.fieldType = BaseFieldTypesEnum.Input
fields.r32.isDisabled = true
fields.r32.variant = "secondary"
fields.r32.dataSource = () => {
    return formStore.ptrData?.results.r32 ?? "";
};
fields.r32.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField(resultFormRegisteredFields.r32, callback);
};
// #endregion R32

// #region Phi 0 Deg
fields.phi0Deg.fieldType = BaseFieldTypesEnum.Input
fields.phi0Deg.isDisabled = true
fields.phi0Deg.variant = "secondary"
fields.phi0Deg.dataSource = () => {
    return formStore.ptrData?.results.phi0_deg ?? "";
};
fields.phi0Deg.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField(resultFormRegisteredFields.phi0Deg, callback);
};
// #endregion Phi 0 Deg

// #region Res Norm
fields.resNorm.fieldType = BaseFieldTypesEnum.Input
fields.resNorm.isDisabled = true
fields.resNorm.variant = "secondary"
fields.resNorm.dataSource = () => {
    return formStore.ptrData?.results.res_norm ?? "";
};
fields.resNorm.deconstructor = (callback: () => void) => {
    return formStore.subscribeToField(resultFormRegisteredFields.resNorm, callback);
};
// #endregion Res Norm

export const resultFormFields = buildFields(fields);