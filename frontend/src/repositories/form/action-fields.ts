import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";

export const actionRegisteredFields = {
    generateButton: "generateButton",
}


const text = lang();
const fields = createFieldPlaceholders(actionRegisteredFields, text.actions);

// #region Generate Button
fields.generateButton.fieldType = BaseFieldTypesEnum.ButtonWithConfirmation
// #endregion Generate Button


export const actionFields = buildFields(fields);