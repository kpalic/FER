package hr.fer.progi.simplicity.services;

import hr.fer.progi.simplicity.entities.Business;
import hr.fer.progi.simplicity.security.exceptions.EntityMissingException;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;

import java.util.List;
import java.util.Optional;

/**
 * Manages business database.
 * @see Business
 */
public interface BusinessService {

    void checkBusinessData(String businessName, String businessAddress, String businessCity, String businessOIB, String businessMobileNumber, String businessDescription);

    /**
     * Lists all businesses.
     * @return iterable containing all businesses
     */
    List<Business> listAll();

    /**
     * Finds business with given OIB, if exists.
     * @param businessOIB given business OIB
     * @return Optional with value of business associated with given ID in the system,
     * or no value if one does not exist
     */
    Optional<Business> findByOIB(String businessOIB);

    /**
     * Creates new business with given name, address, OIB, owner's mobile number and small description.
     * @param businessName name of the new business
     * @param businessAddress location address of the new business
     * @param businessOIB OIB of the new business
     * @param mobileNumber the phone number of the owner of new business
     * @param description small description about the new business
     * @return created Business object, with ID set and a known owner of the business
     * @throws IllegalArgumentException if name, address, OIB, mobile number
     * or description is empty or any is <code>null</code>
     * @throws RequestDeniedException if  address, OIB or mobile number
     * is already a variable that is part of another business
     */
    Business createBusiness(String businessName, String businessType, String businessAddress, String businessCity, String businessOIB, String mobileNumber, String description);

    /**
     * Updates the name of a given business.
     * @param businessId identifies business to update
     * @param name new name of the business
     * @return updated business object
     * @throws EntityMissingException if entity with the same ID as in parameter does not exist
     * @throws IllegalArgumentException if name is empty or any is <code>null</code>
     */
    Business updateBusinessName(long businessId, String name);

    /**
     * Updates the description of a given business.
     * @param businessId identifies business to update
     * @param description new mobile number of the business's owner
     * @return updated business object
     * @throws EntityMissingException if entity with the same ID as in parameter does not exist
     * @throws IllegalArgumentException if description is empty or any is <code>null</code>
     */
    Business updateBusinessDescription(long businessId, String description);

    /**
     * Completely removes business and any information about the business from database
     * @param businessOIB identifies business to delete
     * @return <code>true</code> if business was deleted,
     * <code>false</code> if the business did not exist to begin with.
     * @throws EntityMissingException if an entity with one or the other ID is not found
     * @throws IllegalArgumentException if any argument <code>null</code>
     */
    public void deleteBusiness(String businessOIB);

    public Business setPromoteDuration(String businessOIB, String promoteDuration);
}
