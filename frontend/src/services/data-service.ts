import {createServiceClient} from "@bogunoz/simplify/services";
import type {DataRequestDto} from "../models/data-request-dto.ts";
import {BaseResponseTypeEnum} from "@bogunoz/simplify/models";

// Development localny http://localhost:8000
const api = createServiceClient("http://127.0.0.1:8000");

export interface DataServiceInterface {
    fetchPtrResults: (form: FormData) => Promise<unknown>;
}

class DataService implements DataServiceInterface {
    fetchPtrResultsUrl = "/ptr-fitting";

    fetchPtrResults = async (form: FormData) => {
        return api(this.fetchPtrResultsUrl, {
            method: "POST",
            body: form,

            responseType: BaseResponseTypeEnum.Json
        });
    };

}

export const dataService = new DataService();