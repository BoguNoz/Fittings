import {createServiceClient} from "@bogunoz/simplify/services";
import {BaseResponseTypeEnum} from "@bogunoz/simplify/models";

// Development localny http://localhost:8000
const api = createServiceClient("http://localhost:8000");

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