import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";

export const resultFormRegisteredFields = {
    anisotropy: "anisotropy",
    k2: "k2",
    alfa2: "alfa2r",
    r32: "r32",
    kParallel: "kParallel",
    r2Amp: "r2Amp",
    r2Phase: "r2Phase",
    resNorm: "resNorm",
}


const text = lang();
const fields = createFieldPlaceholders(resultFormRegisteredFields, text.ptrFitResult);

// #region R2 Phase
fields.r2Phase.fieldType = BaseFieldTypesEnum.Input
fields.r2Phase.isDisabled = true
fields.r2Phase.variant = "secondary"
// #endregion R2 Phase


// #region R2 Amp
fields.r2Amp.fieldType = BaseFieldTypesEnum.Input
fields.r2Amp.isDisabled = true
fields.r2Amp.variant = "secondary"
// #endregion R2 Amp


// #region K Parallel
fields.kParallel.fieldType = BaseFieldTypesEnum.Input
fields.kParallel.isDisabled = true
fields.kParallel.variant = "secondary"
// #endregion K Parallel


// #region Anisotropy
fields.anisotropy.fieldType = BaseFieldTypesEnum.Input
fields.anisotropy.isDisabled = true
fields.anisotropy.variant = "secondary"
// #endregion Anisotropy


// #region K2
fields.k2.fieldType = BaseFieldTypesEnum.Input
fields.k2.isDisabled = true
fields.k2.variant = "secondary"
// #endregion K2

// #region Alfa2
fields.alfa2.fieldType = BaseFieldTypesEnum.Input
fields.alfa2.isDisabled = true
fields.alfa2.variant = "secondary"
// #endregion Alfa2

// #region R32
fields.r32.fieldType = BaseFieldTypesEnum.Input
fields.r32.isDisabled = true
fields.r32.variant = "secondary"
// #endregion R32

// #region Res Norm
fields.resNorm.fieldType = BaseFieldTypesEnum.Input
fields.resNorm.isDisabled = true
fields.resNorm.variant = "secondary"
// #endregion Res Norm

export const resultFormFields = buildFields(fields);