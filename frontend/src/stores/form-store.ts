import { BaseStore, autoRegister, type BaseFieldModel, type BaseOperationFn} from "@bogunoz/simplify";
import {toast} from "sonner";
import {lang} from "../text/utils/lang.ts";
import {dataService, type DataServiceInterface} from "../services/data-service.ts";
import type { PTRChartData } from "../models/data-responses.ts";
import {mapPtrResponse} from "../mappers/map-from-response.ts";
import {resultFormRegisteredFields} from "../repositories/form/result-form.ts";

const text = lang();

export interface FormStoreInterface extends BaseStore {
    fetchPtrResults: (form: FormData) => Promise<void>;
}

export class FormStore extends BaseStore {
    override fields: Record<string, BaseFieldModel> = {};
    override operations: Record<string, BaseOperationFn[]> = {};

    service: DataServiceInterface = dataService;
    ptrData: PTRChartData | null = null;

    constructor() {
        super();
        autoRegister(this);
    }

    fetchPtrResults = async (form: FormData): Promise<void> => {
        if (!form) {
            toast.error(text.errors.fetchPtrResultsFailed);
            return;
        }


        try {
            const rawJson = await this.service.fetchPtrResults(form);
            if (!rawJson) {
                toast.error(text.errors.fetchPtrResultsFailed);
                return;
            }
            this.ptrData = mapPtrResponse(rawJson);
            console.log(this.ptrData)

            this.notifyFieldChanged("phase");
            this.notifyFieldChanged("normalizedAmplitudeLog");

            this.setFieldValue(resultFormRegisteredFields.anisotropy, this.ptrData?.results.anisotropy)
            this.setFieldValue(resultFormRegisteredFields.k2, this.ptrData?.results.k2)
            this.setFieldValue(resultFormRegisteredFields.alfa2, this.ptrData?.results.alfa2)
            this.setFieldValue(resultFormRegisteredFields.r32, this.ptrData?.results.r32)
            this.setFieldValue(resultFormRegisteredFields.kParallel, this.ptrData?.results.kParallel)
            this.setFieldValue(resultFormRegisteredFields.r2Amp, this.ptrData?.results.r2Amp)
            this.setFieldValue(resultFormRegisteredFields.r2Phase, this.ptrData?.results.r2Phase)
            this.setFieldValue(resultFormRegisteredFields.resNorm, this.ptrData?.results.resNorm)


        } catch (error) {
            console.error(error);
            toast.error(text.errors.fetchPtrResultsFailed);
        }

    }
}

export const formStore = new FormStore();