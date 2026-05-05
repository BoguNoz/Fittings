import {lang} from "../text/utils/lang.ts";
import type {BaseStore} from "@bogunoz/simplify";

const text = lang();

export const validateThickness = (store: BaseStore, value: any, id: string) => {
    if (value === undefined || value === null || value === "") {
        return { isValid: true, isWarning: false, message: "" };
    }

    const val = parseFloat(value);


    if (val > 0.001) {
        return {
            isValid: true,
            isWarning: true,
            message: text.validators.validateThickness,
        };
    }

    return { isValid: true, isWarning: false, message: "" };
};

export const validateDiffusivity = (store: BaseStore, value: any, id: string) => {
    if (value === undefined || value === null || value === "") {
        return { isValid: true, isWarning: false, message: "" };
    }

    const val = parseFloat(value);


    if (val > 0.01 || val < 1e-10) {
        return {
            isValid: true,
            isWarning: true,
            message: text.validators.validateDiffusivity,
        };
    }

    return { isValid: true, isWarning: false, message: "" };
};