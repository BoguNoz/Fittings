import type {BaseOperationFn} from "@bogunoz/simplify";
import {actionRegisteredFields} from "../repositories/form/action-fields.ts";
import { mapToDataRequest } from "../mappers/map-to-dto.ts";
import type {FormStoreInterface} from "../stores/form-store.ts";
import {ptrRegisteredFields} from "../repositories/form/ptr-fields.ts";

export const sendPtrRequest = (formStore: FormStoreInterface): BaseOperationFn => {
    return async () => {
        const generateButtonId = actionRegisteredFields.generateButton;

        const file = formStore.fields[ptrRegisteredFields.dataInput];
        formStore.validateField(ptrRegisteredFields.dataInput)
        if (file.state.status != "valid"){
            formStore.setFieldState(generateButtonId, "error")
            return;
        }

        formStore.setFieldState(generateButtonId, "pending")
        formStore.setFieldEditability(generateButtonId, false);

        const request = mapToDataRequest(formStore);
        if (!request) {
            return;
        }

        const formData = new FormData();
        formData.append('file', request.file); // To jest obiekt File
        formData.append('k1', request.k1.toString());
        formData.append('l1', request.l1.toString());
        formData.append('l2', request.l2.toString());
        formData.append('alfa1', request.alfa1.toString());
        formData.append('alfa2', request.alfa2.toString());
        formData.append('alfa3', request.alfa3.toString());
        formData.append('r21', request.r21.toString());
        formData.append('sample_name', request.sample_name);
        formData.append('use_hankel', String(request.use_hankel));

        await formStore.fetchPtrResults(formData);

        try {
            formStore.setFieldState(generateButtonId, "valid")
            formStore.setFieldEditability(generateButtonId, true);
        }
        catch {
            formStore.setFieldState(generateButtonId, "error")
            formStore.setFieldEditability(generateButtonId, true);
        }

    }
}