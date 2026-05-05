import {type BaseSectionModel, buildComposites, createCompositesPlaceholders} from "@bogunoz/simplify";
import {SectionCompositeSectionType} from "@bogunoz/simplify/components";
import {configFields} from "./form/config-fields.ts";
import {lang} from "../text/utils/lang.ts";

const text = lang();

export const registeredAppComposites = {
    configSection: "configSection",
}


// #region Config Section
const composites = createCompositesPlaceholders(registeredAppComposites);
composites.formCard.render = true;
composites.formCard.renderFn = () => true;
composites.formCard.sections = [
    {
        type: SectionCompositeSectionType.SECTION,
        fields: configFields,
        title: text.ptrConfig.sectionTitle,
        description: text.ptrConfig.sectionDescription,
        disable: false,

    } as BaseSectionModel,
];
composites.formCard.mode = "vertical-window";
// #endregion Config Section

export const appComposites = buildComposites(composites);