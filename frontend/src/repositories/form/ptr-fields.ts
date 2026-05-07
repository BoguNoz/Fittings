import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";

export const ptrRegisteredFields = {
    dataInput: "dataInput",
    sampleName: "sampleName",
    useFullHankelModel: "useFullHankelModel",
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

// #region Use Full Hankel Model
fields.useFullHankelModel.fieldType = BaseFieldTypesEnum.Switch
fields.useFullHankelModel.variant = "secondary"
// #endregion Use Full Hankel Model

export const ptrFields = buildFields(fields);