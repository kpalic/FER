import {AxiosResponse} from 'axios'
import { createContext, useContext, useState, ReactNode } from 'react'
import { User } from '../models/User.model'
import { AxiosInstance } from './AxiosInstance'

type AuthContextData = {
    user?: User
    loading: boolean
    logIn: (username: string, password: string) => Promise<AxiosResponse>
    registerOwner: (
      username:string,
      email: string,
      password:string,
      businessName:string,
      businessType:string,
      businessAddress:string,
      businessCity:string,
      businessOIB: string,
      businessMobileNumber:string,
      businessDescription:string,
      cardNumber:string,
      expiryDateMonth:string,
      expiryDateYear:string,
      cvv:string
    ) => Promise<AxiosResponse>
    registerUser:(
      username:string,
      email: string,
      password:string
    )=>Promise<AxiosResponse>

    addLocation:(
      longitude:number,
      latitude:number,
      name:string,
      type:string,
      rating:string
    )=>Promise<AxiosResponse>

    changeLocationRating:(
      id:number,
      rating:string
    )=>Promise<AxiosResponse>
    
    logout: () => void
  }

  const AuthContext = createContext<AuthContextData>({} as AuthContextData)

  export const AuthProvider = ({ children }: { children: ReactNode }): JSX.Element => {
    const [user, setUser] = useState<User>()
    const [loading, setLoading] = useState<boolean>(false)
  
    const logIn = (username: string, password: string) => {
      return new Promise<AxiosResponse>((resolve, reject) => {
        setLoading(true)
  
        AxiosInstance.post('/auth/login', {
          username,
          password,
        })
          .then(res => {
            const { data: user } = res
            setUser(user)
            sessionStorage.setItem('token', user.accessToken)
            sessionStorage.setItem('role', JSON.stringify(user.role))
            resolve(user)
          })
          .catch(err => {
            reject(err)
          })
          .finally(() => setLoading(false))
      })
    }
  
    const registerOwner = (
        username:string,
        email: string,
        password:string,
        businessName:string,
        businessType:string,
        businessAddress:string,
        businessCity:string,
        businessOIB: string,
        businessMobileNumber:string,
        businessDescription:string,
        cardNumber:string,
        expiryDateMonth:string,
        expiryDateYear:string,
        cvv:string
    ) => {
      return new Promise<AxiosResponse>((resolve, reject) => {
        setLoading(true)
  
        AxiosInstance.post('/auth/register/owner', {
          username,
          email,
          password,
          businessName,
          businessType,
          businessAddress,
          businessCity,
          businessOIB,
          businessMobileNumber,
          businessDescription,
          cardNumber,
          expiryDateMonth,
          expiryDateYear,
          cvv
        })
          .then(res => {
            resolve(res.data)
          })
          .catch(err => reject(err))
          .finally(() => setLoading(false))
      })
    }

    const registerUser = (
      username:string,
      email: string,
      password:string
  ) => {
    return new Promise<AxiosResponse>((resolve, reject) => {
      setLoading(true)

      AxiosInstance.post('/auth/register/user', {
        username,
        email,
        password
      })
        .then(res => {
          resolve(res.data)
        })
        .catch(err => reject(err))
        .finally(() => setLoading(false))
    })
  }

  const addLocation = (
    longitude: number,
    latitude: number,
    name: string,
    type: string,
    rating: string
  ) => {
    return new Promise<AxiosResponse>((resolve, reject) => {
      setLoading(true);

      AxiosInstance.post("/map", {
        longitude,
        latitude,
        name,
        type,
        rating
      })
        .then((res) => {
          resolve(res.data);
        })
        .catch((err) => reject(err))
        .finally(() => setLoading(false));
    });
  };

  const changeLocationRating=(id:number,rating:string)=>{
    return new Promise<AxiosResponse>((resolve, reject) => {
      setLoading(true);

      AxiosInstance.put("/map", {id,rating})
        .then((res) => {
          resolve(res.data);
        })
        .catch((err) => reject(err))
        .finally(() => setLoading(false));
    });
  }
  
    const logout = () => {
      sessionStorage.removeItem('role')
      sessionStorage.removeItem('token')
      setUser(undefined)
    }
  
    return (
      <AuthContext.Provider value={{ user, loading, logIn, registerOwner, registerUser, addLocation, changeLocationRating, logout }}>
        {children}
      </AuthContext.Provider>
    )
  }

  export const useAuth = (): AuthContextData => {
    const context = useContext(AuthContext)
  
    if (!context) {
      throw new Error('useAuth must be used within an AuthProvider')
    }
  
    return context
  }