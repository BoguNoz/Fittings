import {type BaseCompositeInterface, BaseField, composite} from "@bogunoz/simplify";
import {actionRegisteredFields} from "../../../repositories/form/action-fields.ts";

const ActionComposite = composite((props : BaseCompositeInterface) => {
    const {compositeId, compositeStore, store, handleBlur, handleChange,} = props;

    return(
        <div className="flex justify-end w-full pr-8">
            <BaseField
                fieldId={actionRegisteredFields.generateButton}
                store={store}
                handleBlur={handleBlur}
                handleChange={handleChange}
            />
        </div>
    )
})

export default ActionComposite;