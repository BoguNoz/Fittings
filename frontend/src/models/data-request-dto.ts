export interface DataRequestDto {
    l2: number;
    k1: number;
    l1: number;
    alfa1: number;
    alfa2: number;
    alfa3: number;
    r21: number;
    weight: number;

    sample_name: string;
    use_hankel: boolean;
    file: File;
}
