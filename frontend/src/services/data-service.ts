import {createServiceClient} from "@bogunoz/simplify/services";
import type {DataRequestDto} from "../models/data-request-dto.ts";
import {BaseResponseTypeEnum} from "@bogunoz/simplify/models";

// Development localny http://localhost:8000
const api = createServiceClient("http://localhost:8000");

export interface DataServiceInterface {
    fetchPtrResults: (dataRequest: DataRequestDto) => Promise<ArrayBuffer>;
}

class DataService implements DataServiceInterface {
    fetchPtrResultsUrl = "/ptr-results";

    fetchPtrResults = async (dataRequest: DataRequestDto) => {
        return api<ArrayBuffer>(this.fetchPtrResultsUrl, {
            method: "POST",
            body: JSON.stringify(dataRequest),
            responseType: BaseResponseTypeEnum.ArrayBuffer
        });
    };

}

export const dataService = new DataService();