import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";
import {isGreaterThenZero, isNumber, isPositive} from "@bogunoz/simplify/events";
import {validateDiffusivity, validateThickness} from "../../events/validators.ts";

export const configRegisteredFields = {
    l2: "l2",
    k1: "k1",
    l1: "l1",
    alfa1: "alfa1",
    alfa2: "alfa2",
    alfa3: "alfa3",
    r21: "r21",
    weight: "weight",
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
fields.l2.addit!.placeholder = "469e-9"
// #endregion L2

// #region K1
fields.k1.fieldType = BaseFieldTypesEnum.Input
fields.k1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.k1.addit!.placeholder = "21.0"
// #endregion K1

// #region L1
fields.l1.fieldType = BaseFieldTypesEnum.Input
fields.l1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateThickness
];
fields.l1.addit!.placeholder = "80e-9"
// #endregion L1

// #region Alfa1
fields.alfa1.fieldType = BaseFieldTypesEnum.Input
fields.alfa1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
fields.alfa1.addit!.placeholder = "8.9e-6"
// #endregion Alfa1

// #region Alfa2
fields.alfa2.fieldType = BaseFieldTypesEnum.Input
fields.alfa2.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
// #endregion Alfa1

// #region Alfa3
fields.alfa3.fieldType = BaseFieldTypesEnum.Input
fields.alfa3.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
fields.alfa3.addit!.placeholder = "6.0e-6"
// #endregion Alfa3

// #region R21
fields.r21.fieldType = BaseFieldTypesEnum.Input
fields.r21.validators = [
    isNumber,
    isPositive,
];
fields.r21.addit!.placeholder = "2.8e-8"
// #endregion R21

// #region R21
fields.weight.fieldType = BaseFieldTypesEnum.Input
fields.weight.validators = [
    isNumber,
    isPositive,
];
fields.weight.addit!.placeholder = "3.3"
// #endregion R21


export const configFields = buildFields(fields);