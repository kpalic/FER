export enum LocationType {
    BEACH='BEACH',
    PARK='PARK',
    RESTAURANT='RESTAURANT',
    COFFEE_BAR='COFFEE_BAR',
    OTHER='OTHER'
  }

export type Location = {
    id:number
    latitude:number
    longitude:number
    name:string
    type: LocationType
    votesSum:number
    positiveVotes:number
    rating:number
  }