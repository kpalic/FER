import { useState, useEffect} from 'react'
import { AxiosInstance } from '../../auth/AxiosInstance'

export const Ratings = ({ratings}: {ratings:any}) => { 
    
    const [data, setData] = useState<any[]>([])
    const [updateState, setUpdateState] = useState(false)
    const [deleteState, setDeleteState] = useState(false)
    const [change, setChange] = useState(false)
    const [ratingId, setRatingId] = useState<string>("")
    const [ratingScore, setRatingScore] = useState<string>("")
    const [check, setCheck] = useState(true)

    const changeRating = (ratingId : string, ratingType : string) => {
        setRatingId(ratingId)
        setRatingScore(ratingType)
        AxiosInstance.put('profile/user/editRating', {
            ratingId,
            ratingType
        }).then(() => {
            setUpdateState(true)
            setChange(!change)
        }).catch(() => {
        })
    }

    const deleteRating = (ratingId : string) => {
        setRatingId(ratingId)
        AxiosInstance.delete('profile/user/ratingDelete?ratingId=' + ratingId, {
        }).then(() => {
            setDeleteState(true)
            setChange(!change)
        }).catch(() => {
        })
    }

    useEffect(()=> {
        if (deleteState === true) {
            for (let i = 0; i < ratings.length; ++i) {
                if (ratings[i].ratingId === ratingId) {
                    ratings.splice(i, 1)
                    break
                }
            }
        }

        if (updateState === true) {
            for (let i = 0; i < ratings.length; ++i) {
                if (ratings[i].ratingId === ratingId) {
                    if (ratingScore === "POSITIVE") {
                        ratings[i].ratingType = 'POSITIVE'
                        break;
                    } else {
                        ratings[i].ratingType = 'NEGATIVE'
                        break;
                    }
                }
            }
        }
        if (ratings.length === 0) {
            setCheck(false)
        }
        setData(ratings)
        setDeleteState(false)
        setUpdateState(false)
    }, [change]);


    return (
        <div className='ratings'>
            {check ? (<>
            <div className='user-ratings-title'>
                <h1>Ocjene lokacija</h1>
            </div>
            
            <table className='table-rating'>
                <tbody>
                    <tr>
                        <th className='column-rating-name'>
                            Ime lokacije
                        </th>
                        <th className='column-rating-name'>
                            Prikladna
                        </th>
                        <th className='column-rating-name'>
                            Neprikladna
                        </th>
                        <th className='column-rating-name'>
                            Obriši ocjenu
                        </th>
                    </tr>
                    
                    {(data as any[]).map((value) => (
                        <tr className='table-row' key={value.ratingId}>
                            <th className='data-rating-text'>
                                {value.locationName}
                            </th>
                            <th className='data-rating-edit'>
                                {value.ratingType ===  'POSITIVE' ? 
                                    (<img className='icons-edit-active' src={require('./../../assets/icons/likeColor2.png')} alt='like'></img>)
                                    :
                                    (<img className='icons-edit' onClick={() => changeRating(value.ratingId, "POSITIVE")} 
                                    src={require('./../../assets/icons/like.png')} alt='like'></img>) 
                                }
                            </th>
                            <th className='data-rating-edit'>
                                {value.ratingType === 'NEGATIVE' ? 
                                    (<img className='icons-edit-active' src={require('./../../assets/icons/dislikeColor.png')} alt='dislike'></img>)
                                    :
                                    (<img className='icons-edit' onClick={() => changeRating(value.ratingId, "NEGATIVE")} 
                                    src={require('./../../assets/icons/dislike.png')} alt='dislike'></img>) 
                                }
                            </th>
                            <th className='data-rating-edit'>
                                <img className='icons-edit-delete' onClick={() => {
                                    const confirmBox = window.confirm(
                                        "Želite li zaista izbrisati svoju ocjenu za lokaciju " + value.locationName + "?"
                                    )
                                    if (confirmBox === true) {
                                        deleteRating(value.ratingId)
                                    }
                                }} 
                                src={require('./../../assets/icons/delete.png')} alt='delete'></img>
                            </th>
                        </tr>
                    ))}
                </tbody>
            </table>
            </>) : (
            <div className='no-rating'>
                <h1>Niste dodijelili niti jednu prikladnost lokacijama</h1>
            </div>)}
        </div>
    )
}