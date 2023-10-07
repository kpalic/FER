export enum UserRole {
    ROLE_USER= 'USER',
    ROLE_OWNER = 'OWNER',
  }
  
  export type User = {
    accessToken: string
    id:number
    role:UserRole
    username:string
  }
  