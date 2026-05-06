import { observer } from "mobx-react-lite";

import {
    type BaseCompositeStore,
    type BaseStore, SectionComposite, SheetComposite,
} from "@bogunoz/simplify";
import {registeredAppComposites} from "../../repositories/composites.ts";
import ActionComposite from "./partials/ActionComposite.tsx";


export interface FormSidePanelProps {
    compositeStore: BaseCompositeStore;
    formStore: BaseStore;
    handleBlur?: (fieldId: string) => void;
    handleChange?: (fieldId: string, value: any) => void;
}


const FormSidePanel = observer((props: FormSidePanelProps) => {
    return (
        <SheetComposite
            compositeId={registeredAppComposites.sheet}
            compositeStore={props.compositeStore}
            store={props.formStore}
        >
            <>
                <ActionComposite
                    compositeId={registeredAppComposites.actions}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                />
                <SectionComposite
                    compositeId={registeredAppComposites.ptrSection}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                    isClosed={true}
                />
                <SectionComposite
                    compositeId={registeredAppComposites.configSection}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                    isClosed={true}
                />
            </>
        </SheetComposite>
    )
});

export default FormSidePanel;