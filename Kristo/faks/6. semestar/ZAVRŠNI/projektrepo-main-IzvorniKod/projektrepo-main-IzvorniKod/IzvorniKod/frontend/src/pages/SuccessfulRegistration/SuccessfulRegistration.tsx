import * as React from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import './SuccessfulRegistration.css'
import { AxiosInstance } from '../../auth/AxiosInstance';

const SuccessfulRegistration=()=>{

    const [queryParameters]=useSearchParams()
    const [username,setUsername]=React.useState("")

      React.useEffect(()=>{
        setUsername(queryParameters.get("username")?.toString()!)
        AxiosInstance.post('/auth/email-confirm', {
            username
          })
            .then(res => {
                console.log(res)
            })
            .catch(err => {
                console.error(err)
            })

      })

    return(<>
    <div className='success-container'>
        <div className='background-container'>
            <div className='registration-container'>
                <div className='text-container-sr'>
                    <div className='t-c'>
                    <b>{username}</b> uspješno ste se registrirali!
                    </div>

                <Link className='go-to-link-sr' to="/auth/login" style={{ textDecoration: 'none', color: 'black' }}>
                  Prijavite se!
                </Link>
                </div>
            </div>
        </div>
    </div>
    </>)
}

export default SuccessfulRegistration;