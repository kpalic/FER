import axios from 'axios'
//https://dogfriendly-webservice-fc8h.onrender.com/api/
//https://dogfriendly-backservice.onrender.com/api/
export const AxiosInstance = axios.create({ baseURL: 'https://dogfriendly-webservice-fc8h.onrender.com/api/' })

AxiosInstance.interceptors.request.use(async request => {
  const token = sessionStorage.getItem('token')

  if (token && request.headers) {
    request.headers['Authorization'] = 'Bearer ' + token
  }

  // TODO else if token is not present

  return request
})