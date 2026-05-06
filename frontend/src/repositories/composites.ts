import {
    type BaseSectionModel,
    buildComposites, ChartCompositeSectionType,
    createCompositesPlaceholders, FormCardCompositeSectionType, SheetCompositeSectionType,
} from "@bogunoz/simplify";
import {SectionCompositeSectionType} from "@bogunoz/simplify/components";
import {configFields} from "./form/config-fields.ts";
import {lang} from "../text/utils/lang.ts";
import {ptrFields} from "./form/ptr-fields.ts";
import {actionFields} from "./form/action-fields.ts";
import {amplitudeLinearField, normalizedAmplitudeLogField, phaseField} from "./dashboard/chart-fields.ts";
import {resultFormFields} from "./form/result-form.ts";

const text = lang();

export const registeredAppComposites = {
    actions: "actions",
    sheet: "sheet",
    ptrSection: "ptrSection",
    configSection: "configSection",

    amplitudeLinearChart: "amplitudeLinearChart",
    normalizedAmplitudeLogChart: "normalizedAmplitudeLogChart",
    phaseChart: "phaseChart",
    lineChartCard: "lineChartCard",
    resultForm: "resultForm",
}

const composites = createCompositesPlaceholders(registeredAppComposites);

// #region Actions
composites.actions.render = true;
composites.actions.renderFn = () => true;
composites.actions.sections = [
    {
        fields: actionFields,

    } as BaseSectionModel,
];
// #endregion Actions

// #region Sheet
composites.sheet.render = true;
composites.sheet.renderFn = () => true;
composites.sheet.sections = [
    {
        type: SheetCompositeSectionType.HEADER,
        title: text.ptrExecution.sectionTitle,
        description: text.ptrExecution.sectionDescription,
        disable: false,

    } as BaseSectionModel,
];
composites.sheet.mode = "vertical-window";
composites.sheet.size = 0.7;
// #endregion Sheet

// #region PTR Section
composites.ptrSection.render = true;
composites.ptrSection.renderFn = () => true;
composites.ptrSection.sections = [
    {
        type:  SectionCompositeSectionType.SECTION,
        fields: ptrFields,
        title: text.ptr.sectionTitle,
        description: text.ptr.sectionDescription,
        disable: false,

    } as BaseSectionModel,
];
composites.ptrSection.mode = "vertical-window";
// #endregion PTR Section

// #region Config Section
composites.configSection.render = true;
composites.configSection.renderFn = () => true;
composites.configSection.sections = [
    {
        type: SectionCompositeSectionType.SECTION,
        fields: configFields,
        title: text.ptrConfig.sectionTitle,
        description: text.ptrConfig.sectionDescription,
        disable: false,

    } as BaseSectionModel,
];
composites.configSection.mode = "square-window";
// #endregion Config Section

// #region Amplitude Linear Chart
composites.amplitudeLinearChart.render = true;
composites.amplitudeLinearChart.renderFn = () => true;
composites.amplitudeLinearChart.sections = [
    {
        type: ChartCompositeSectionType.HEADER,
        title: text.ptrPlots.sectionTitle,
        description: text.ptrPlots.sectionDescription,
        disable: false,

    } as BaseSectionModel,
    {
        type: ChartCompositeSectionType.LINE_CHART,
        fields: amplitudeLinearField,
        disable: false,

    } as BaseSectionModel,
];
composites.amplitudeLinearChart.mode = "horizontal-window";
composites.amplitudeLinearChart.size = 0.65;
// #endregion Amplitude Linear Chart

// #region Normalized Amplitude Log Chart
composites.normalizedAmplitudeLogChart.render = true;
composites.normalizedAmplitudeLogChart.renderFn = () => true;
composites.normalizedAmplitudeLogChart.sections = [
    {
        type: ChartCompositeSectionType.HEADER,
        title: text.ptrPlots.sectionTitle,
        description: text.ptrPlots.sectionDescription,
        disable: false,

    } as BaseSectionModel,
    {
        type: ChartCompositeSectionType.LINE_CHART,
        fields: normalizedAmplitudeLogField,
        disable: false,

    } as BaseSectionModel,
];
composites.normalizedAmplitudeLogChart.mode = "horizontal-window";
composites.normalizedAmplitudeLogChart.size = 0.65;
// #endregion Normalized Amplitude Log Chart

// #region Phase Chart
composites.phaseChart.render = true;
composites.phaseChart.renderFn = () => true;
composites.phaseChart.sections = [
    {
        type: ChartCompositeSectionType.HEADER,
        title: text.ptrPlots.sectionTitle,
        description: text.ptrPlots.sectionDescription,
        disable: false,

    } as BaseSectionModel,
    {
        type: ChartCompositeSectionType.LINE_CHART,
        fields: phaseField,
        disable: false,

    } as BaseSectionModel,
];
composites.phaseChart.mode = "horizontal-window";
composites.phaseChart.size = 0.65;
// #endregion Phase Chart

// #region Line Chart Card
composites.lineChartCard.render = true;
composites.lineChartCard.renderFn = () => true;
composites.lineChartCard.mode = "square-window";
// #endregion Line Chart Card

// #region Result Form
composites.resultForm.render = true;
composites.resultForm.renderFn = () => true;
composites.resultForm.sections = [
    {
        type: ChartCompositeSectionType.HEADER,
        title: text.ptrFitResult.sectionTitle,
        description: text.ptrFitResult.sectionDescription,
        disable: false,

    } as BaseSectionModel,
    {
        type: FormCardCompositeSectionType.BODY,
        fields: resultFormFields,
        disable: false,

    } as BaseSectionModel,
];
composites.resultForm.mode = "vertical-window";
composites.resultForm.size = 0.65;
// #endregion Result Form

export const appComposites = buildComposites(composites);