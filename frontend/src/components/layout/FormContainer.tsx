import type { BaseCompositeStore, BaseStore } from "@bogunoz/simplify";

export interface FormContainerProps {
    compositeId?: string;

    compositeStore: BaseCompositeStore;
    fieldStore: BaseStore;
}

