import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, isInteger,} from "@bogunoz/simplify";
import {isPositive} from "@bogunoz/simplify/events";

export const ptrRegisteredFields = {
    dataInput: "dataInput",
    sampleName: "sampleName",
    nStarts: "nStarts",
}


const text = lang();
const fields = createFieldPlaceholders(ptrRegisteredFields, text.ptr);

// #region Data Input
fields.dataInput.fieldType = BaseFieldTypesEnum.FileInput
fields.dataInput.isRequired = true;
// #endregion Data Input

// #region Sample Name
fields.sampleName.fieldType = BaseFieldTypesEnum.Input
fields.sampleName.variant = "secondary"
// #endregion Sample Name

// #region N Starts
fields.nStarts.fieldType = BaseFieldTypesEnum.Input
fields.nStarts.validators = [
    isInteger,
    isPositive,
];
fields.nStarts.addit!.placeholder = "20"
// #endregion N Starts

export const ptrFields = buildFields(fields);