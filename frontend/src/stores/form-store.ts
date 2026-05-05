import { BaseStore, autoRegister, type BaseFieldModel, type BaseOperationFn} from "@bogunoz/simplify";


export class FormStore extends BaseStore {
    override fields: Record<string, BaseFieldModel> = {};
    override operations: Record<string, BaseOperationFn[]> = {};

    constructor() {
        super();
        autoRegister(this);
    }
}

export const formStore = new FormStore();