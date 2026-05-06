import {type BaseCompositeStore, type BaseStore, computeCompositeSize, MetadataContext, type MetadataModel} from "@bogunoz/simplify";
import {observer} from "mobx-react-lite";
import {ChartComposite} from "@bogunoz/simplify/components";
import {registeredAppComposites} from "../../repositories/composites.ts";
import LineChartCard from "./partials/LineChartCard.tsx";

export interface DashboardProps {
    compositeStore: BaseCompositeStore;
    formStore: BaseStore;
    handleBlur?: (fieldId: string) => void;
    handleChange?: (fieldId: string, value: any) => void;
}


const Dashboard = observer((props: DashboardProps) => {
    /*const [innerWidth, innerHeight] = computeCompositeSize("square-window", 1.0);
    const metadata = {
        width: innerWidth ,
        height: innerHeight,
    } as MetadataModel;*/

    return (
        <LineChartCard
            compositeId={registeredAppComposites.lineChartCard}
            compositeStore={props.compositeStore}
            store={props.formStore}
        />
    )

   /* return (
        //<MetadataContext.Provider value={metadata}>
            <div>
                <ChartComposite
                    compositeId={registeredAppComposites.amplitudeLinearChart}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                    labels={["Model", "Experiment"]}
                    palette={["#4f46e5", "#db2777"]}
                    legends={true}
                />

                <ChartComposite
                    compositeId={registeredAppComposites.normalizedAmplitudeLogChart}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                    labels={["Model", "Experiment"]}
                    palette={["#4f46e5", "#db2777"]}
                    legends={true}
                />

                <ChartComposite
                    compositeId={registeredAppComposites.phaseChart}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                    labels={["Model", "Experiment"]}
                    palette={["#4f46e5", "#db2777"]}
                    legends={true}
                />

            </div>
       // </MetadataContext.Provider>

    )*/
});

export default Dashboard;