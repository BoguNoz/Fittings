import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";
import {isGreaterThenZero, isNumber, isPositive} from "@bogunoz/simplify/events";
import {validateDiffusivity, validateThickness} from "../../events/validators.ts";

export const configRegisteredFields = {
    l2: "l2",
    k1: "k1",
    l1: "l1",
    alfa1: "alfa1",
    k3: "k3",
    alfa3: "alfa3",
    r21: "r21",
    dPump: "dPump",
    anisotropy: "anisotropy",
    q: "q",
    rhoc2: "rhoc2",
    weight: "weight",
}

const text = lang();
const fields = createFieldPlaceholders(configRegisteredFields, text.ptrConfig);

// #region RHOC2
fields.rhoc2.fieldType = BaseFieldTypesEnum.Input
fields.rhoc2.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.rhoc2.addit!.placeholder = "1.3e6"
// #endregion RHOC2

// #region Anisotropy
fields.anisotropy.fieldType = BaseFieldTypesEnum.Input
fields.anisotropy.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.anisotropy.addit!.placeholder = "1.94"
// #endregion Anisotropy

// #region Q
fields.q.fieldType = BaseFieldTypesEnum.Input
fields.q.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.q.addit!.placeholder = "1.0"
// #endregion Q


// #region D PUMP
fields.dPump.fieldType = BaseFieldTypesEnum.Input
fields.dPump.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.dPump.addit!.placeholder = "2.40e-6"
// #endregion D PUMP


// #region L2
fields.l2.fieldType = BaseFieldTypesEnum.Input
fields.l2.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateThickness,
];
fields.l2.addit!.placeholder = "240e-9"
// #endregion L2

// #region K1
fields.k1.fieldType = BaseFieldTypesEnum.Input
fields.k1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.k1.addit!.placeholder = "150.0"
// #endregion K1

// #region K3
fields.k3.fieldType = BaseFieldTypesEnum.Input
fields.k3.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
];
fields.k3.addit!.placeholder = "1.0"
// #endregion K3

// #region L1
fields.l1.fieldType = BaseFieldTypesEnum.Input
fields.l1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateThickness
];
fields.l1.addit!.placeholder = "50e-9"
// #endregion L1

// #region Alfa1
fields.alfa1.fieldType = BaseFieldTypesEnum.Input
fields.alfa1.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
fields.alfa1.addit!.placeholder = "2.1e-5"
// #endregion Alfa1

// #region Alfa3
fields.alfa3.fieldType = BaseFieldTypesEnum.Input
fields.alfa3.validators = [
    isNumber,
    isPositive,
    isGreaterThenZero,
    validateDiffusivity
];
fields.alfa3.addit!.placeholder = "0.5e-7"
// #endregion Alfa3

// #region R21
fields.r21.fieldType = BaseFieldTypesEnum.Input
fields.r21.validators = [
    isNumber,
    isPositive,
];
fields.r21.addit!.placeholder = "1.0e-8"
// #endregion R21

// #region Weight
fields.weight.fieldType = BaseFieldTypesEnum.Input
fields.weight.validators = [
    isNumber,
    isPositive,
];
fields.weight.addit!.placeholder = "1"
// #endregion Weight


export const configFields = buildFields(fields);