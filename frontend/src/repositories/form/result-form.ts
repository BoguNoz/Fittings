import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";

export const resultFormRegisteredFields = {
    k2: "k2",
    alfa2r: "alfa2r",
    r32: "r32",
    phi0Deg: "phi0Deg",
    resNorm: "resNorm",
}


const text = lang();
const fields = createFieldPlaceholders(resultFormRegisteredFields, text.ptrFitResult);

// #region K2
fields.k2.fieldType = BaseFieldTypesEnum.Input
fields.k2.isDisabled = true
fields.k2.variant = "secondary"
// #endregion K2

// #region Alfa2
fields.alfa2r.fieldType = BaseFieldTypesEnum.Input
fields.alfa2r.isDisabled = true
fields.alfa2r.variant = "secondary"
// #endregion Alfa2

// #region R32
fields.r32.fieldType = BaseFieldTypesEnum.Input
fields.r32.isDisabled = true
fields.r32.variant = "secondary"
// #endregion R32

// #region Phi 0 Deg
fields.phi0Deg.fieldType = BaseFieldTypesEnum.Input
fields.phi0Deg.isDisabled = true
fields.phi0Deg.variant = "secondary"
// #endregion Phi 0 Deg

// #region Res Norm
fields.resNorm.fieldType = BaseFieldTypesEnum.Input
fields.resNorm.isDisabled = true
fields.resNorm.variant = "secondary"
// #endregion Res Norm

export const resultFormFields = buildFields(fields);