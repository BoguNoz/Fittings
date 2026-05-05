import {autoRegister, BaseCompositeStore, BaseStore, type BaseCompositeModel } from "@bogunoz/simplify";

class CompositeStore extends BaseCompositeStore {
    composites: Record<string, BaseCompositeModel> = {};
    stores: Record<string, BaseStore> = {}

    constructor() {
        super();
        autoRegister(this)
    }
}

export const compositeStore = new CompositeStore();