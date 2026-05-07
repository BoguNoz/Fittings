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
            this.notifyFieldChanged("amplitudeLinear");
            this.notifyFieldChanged("normalizedAmplitudeLog");

            this.notifyFieldChanged(resultFormRegisteredFields.k2)
            this.notifyFieldChanged(resultFormRegisteredFields.k3)
            this.notifyFieldChanged(resultFormRegisteredFields.alfa2r)
            this.notifyFieldChanged(resultFormRegisteredFields.r32)
            this.notifyFieldChanged(resultFormRegisteredFields.phi0Deg)
            this.notifyFieldChanged(resultFormRegisteredFields.resNorm)


        } catch (error) {
            console.error(error);
            toast.error(text.errors.fetchPtrResultsFailed);
        }

    }
}

export const formStore = new FormStore();