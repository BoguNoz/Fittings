import {type BaseStore, isNullEmptyFalseOrUndefined} from "@bogunoz/simplify";
import type {DataRequestDto} from "../models/data-request-dto.ts";
import {configRegisteredFields} from "../repositories/form/config-fields.ts";
import {ptrRegisteredFields} from "../repositories/form/ptr-fields.ts";


export const mapToDataRequest = (store: BaseStore) => {
    const getNumberOrDefault = (
        field: any,
        defaultValue: number
    ): number => {
        const value = store.getFieldValue(field);

        return Number(
            isNullEmptyFalseOrUndefined(value)
                ? defaultValue
                : value
        );
    };


    return {
        k1: getNumberOrDefault(configRegisteredFields.k1, 21.0),
        l1: getNumberOrDefault(configRegisteredFields.l1, 80e-9),
        l2: getNumberOrDefault(configRegisteredFields.l2, 469e-9),
        alfa1: getNumberOrDefault(configRegisteredFields.alfa1, 8.9e-6),
        alfa2: getNumberOrDefault(configRegisteredFields.alfa2, -1),
        alfa3: getNumberOrDefault(configRegisteredFields.alfa3, 6.0e-6),
        r21: getNumberOrDefault(configRegisteredFields.r21, 2.8e-8),

        sample_name: store.getFieldValue(ptrRegisteredFields.sampleName) ?? "EXPERIMENT",
        use_hankel: Boolean(store.getFieldValue(ptrRegisteredFields.useFullHankelModel)),
        file: store.getFieldValue(ptrRegisteredFields.dataInput),
    } as DataRequestDto;
};
