import {type BaseCompositeStore, type BaseStore} from "@bogunoz/simplify";
import {observer} from "mobx-react-lite";
import {registeredAppComposites} from "../../repositories/composites.ts";
import LineChartCard from "./partials/LineChartCard.tsx";
import {FormCardComposite} from "@bogunoz/simplify/components";

export interface DashboardProps {
    compositeStore: BaseCompositeStore;
    formStore: BaseStore;
    handleBlur?: (fieldId: string) => void;
    handleChange?: (fieldId: string, value: any) => void;
}


const Dashboard = observer((props: DashboardProps) => {
    return (
        <div className="flex flex-col md:flex-row w-full gap-6 p-4">
            <div className="w-full md:w-1/3">
                <FormCardComposite
                    compositeId={registeredAppComposites.resultForm}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                />
            </div>

            <div className="w-full md:w-2/3">
                <LineChartCard
                    compositeId={registeredAppComposites.lineChartCard}
                    compositeStore={props.compositeStore}
                    store={props.formStore}
                />
            </div>
        </div>
    );
});

export default Dashboard;