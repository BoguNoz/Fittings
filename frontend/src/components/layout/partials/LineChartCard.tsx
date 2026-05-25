import {
    type BaseCompositeInterface,
    Card, CardContent,
    composite,
    type MetadataModel,
    ScrollArea,
    useMetadata
} from "@bogunoz/simplify";
import {ChartComposite} from "@bogunoz/simplify/components";
import {registeredAppComposites} from "../../../repositories/composites.ts";

const LineChartCard = composite((props: BaseCompositeInterface) => {
    const { compositeId, compositeStore, store } = props;

    const metadata = useMetadata() ?? ({} as MetadataModel);

    const composite = compositeStore.composites[compositeId];
    if (!composite) {
        return null;
    }

    const cardStyle = {
        width: `${metadata.width * 0.8}px`,
        height: `${metadata.height * 1.1}px`,
    };

    return (
        <Card style={cardStyle} className="overflow-hidden flex flex-col">
            <ScrollArea className="h-full w-full rounded-md border-none">
                <CardContent className="flex flex-col items-center gap-8 p-6">
                    <ChartComposite
                        compositeId={registeredAppComposites.normalizedAmplitudeLogChart}
                        compositeStore={props.compositeStore}
                        store={props.store}
                        labels={["Model", "Experiment"]}
                        palette={["#4f46e5", "#db2777"]}
                        legends={true}
                    />

                    <ChartComposite
                        compositeId={registeredAppComposites.phaseChart}
                        compositeStore={props.compositeStore}
                        store={props.store}
                        labels={["Model", "Experiment"]}
                        palette={["#4f46e5", "#db2777"]}
                        legends={true}
                    />
                </CardContent>
            </ScrollArea>
        </Card>
    );
});

export default LineChartCard;