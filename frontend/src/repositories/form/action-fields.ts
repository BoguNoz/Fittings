import {lang} from "../../text/utils/lang.ts";
import {BaseFieldTypesEnum, buildFields, createFieldPlaceholders, } from "@bogunoz/simplify";
import {sendPtrRequest} from "../../events/operations.ts";
import {formStore} from "../../stores/form-store.ts";

export const actionRegisteredFields = {
    generateButton: "generateButton",
}


const text = lang();
const fields = createFieldPlaceholders(actionRegisteredFields, text.actions);

// #region Generate Button
fields.generateButton.fieldType = BaseFieldTypesEnum.StatusButton
fields.generateButton.operations = [
    sendPtrRequest(formStore),
]
// #endregion Generate Button


export const actionFields = buildFields(fields);