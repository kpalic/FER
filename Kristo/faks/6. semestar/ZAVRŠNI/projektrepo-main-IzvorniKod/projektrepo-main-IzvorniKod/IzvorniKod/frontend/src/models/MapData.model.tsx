import { Location } from "./Location.model";
import { Business } from "./Business.model";

export type MapData={
    locations:Location[],
    business:Business[],
    positiveRatedLocations: Location[],
    negativeRatedLocations: Location[]
}