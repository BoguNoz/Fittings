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
        l1: getNumberOrDefault(configRegisteredFields.l1, 50e-9),
        k1: getNumberOrDefault(configRegisteredFields.k1, 150.0),
        alfa1: getNumberOrDefault(configRegisteredFields.alfa1, 2.1e-5),
        l2: getNumberOrDefault(configRegisteredFields.l2, 240e-9),
        rhoc2: getNumberOrDefault(configRegisteredFields.rhoc2, 1.3e-6),
        alfa3: getNumberOrDefault(configRegisteredFields.alfa3, 0.5e-7),
        k3: getNumberOrDefault(configRegisteredFields.k3, 1),
        r21: getNumberOrDefault(configRegisteredFields.r21, 1.0e-8),
        d_pump: getNumberOrDefault(configRegisteredFields.dPump, 2.40e-6),
        Q: getNumberOrDefault(configRegisteredFields.q, 1.0),
        anisotropy: getNumberOrDefault(configRegisteredFields.anisotropy, 1.94),
        weight: getNumberOrDefault(configRegisteredFields.weight, 3.3),
        sample_name: store.getFieldValue(ptrRegisteredFields.sampleName) ?? "EXPERIMENT",
        file: store.getFieldValue(ptrRegisteredFields.dataInput),
        n_starts: getNumberOrDefault(ptrRegisteredFields.nStarts, 20),
    } as DataRequestDto;
};
