import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";
import {isGreaterThenZero, isNumber, isPositive} from "@bogunoz/simplify/events";
import {validateDiffusivity, validateThickness} from "../../events/validators.ts";

export const configRegisteredFields = {
    l2: "l2",
    k1: "k1",
    l1: "l1",
    alfa1: "alfa1",
    alfa3: "alfa3",
    r21: "r21",
}


const text = lang();
const fields = createFieldPlaceholders(configRegisteredFields, text.ptrConfig);

// #region L2
fields.l2.fieldType = BaseFieldTypesEnum.Input
fields.l2.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateThickness,
];
fields.l2.variant = "secondary"
// #endregion L2

// #region K1
fields.k1.fieldType = BaseFieldTypesEnum.Input
fields.k1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.k1.variant = "secondary"
// #endregion K1

// #region L1
fields.l1.fieldType = BaseFieldTypesEnum.Input
fields.l1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateThickness
];
fields.l1.variant = "secondary"
// #endregion L1

// #region Alfa1
fields.alfa1.fieldType = BaseFieldTypesEnum.Input
fields.alfa1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
fields.alfa1.variant = "secondary"
// #endregion Alfa1

// #region Alfa3
fields.alfa3.fieldType = BaseFieldTypesEnum.Input
fields.alfa3.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
fields.alfa3.variant = "secondary"
// #endregion Alfa3

// #region R21
fields.r21.fieldType = BaseFieldTypesEnum.Input
fields.r21.validators = [
    isNumber,
    isPositive,
];
fields.r21.variant = "secondary"
// #endregion R21


export const configFields = buildFields(fields);