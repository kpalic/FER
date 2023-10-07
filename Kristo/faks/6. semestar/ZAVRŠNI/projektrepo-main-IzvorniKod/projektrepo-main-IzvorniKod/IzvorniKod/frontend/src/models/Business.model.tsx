export enum BusinessType{
    SHOP='SHOP',
    SALON='SALON',
    VET='VET',
    DAYCARE='DAYCARE',
    OTHER='OTHER'
}
export type Business={
    id:number,
    businessName:string,
    businessType:BusinessType,
    businessAddress:string,
    businessMobileNumber:string,
    businessDescription:string,
    businessCity:string,
    promotionDuration:string,
    promotionStart:string,
}