package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.entities.Business;
import hr.fer.progi.simplicity.entities.BusinessType;
import hr.fer.progi.simplicity.repositories.BusinessRepository;
import hr.fer.progi.simplicity.services.BusinessService;
import hr.fer.progi.simplicity.security.exceptions.EntityMissingException;
import hr.fer.progi.simplicity.security.exceptions.RequestDeniedException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;

@Service
public class BusinessServiceJpa implements BusinessService {
    private static final String businessOIBFormat = "[0-9]{11}";
    private static final Pattern businessPhoneNumberFormat = Pattern.compile("^\\+?\\d{1,4}?[-./\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}$");

    @Autowired
    private BusinessRepository businessRepository;

    @Override
    public List<Business> listAll() {
        return businessRepository.findAll();
    }

    @Override
    public Optional<Business> findByOIB(String OIB) {
        return businessRepository.findByBusinessOIB(OIB);
    }

    @Override
    public Business createBusiness(String businessName, String businessType, String businessAddress, String businessCity, String businessOIB, String mobileNumber, String description) {
        BusinessType type = BusinessType.valueOf(businessType.toUpperCase());
        return businessRepository.save(new Business(businessName, type, businessAddress, businessCity, businessOIB, mobileNumber, description));
    }

    @Override
    public Business updateBusinessName(long businessId, String businessName) {
        //CHECK DATA
        Assert.notNull(businessName, "Ime obrta mora biti predano.");
        Assert.hasText(businessName, "Ime obrta mora biti postavljeno.");
        Assert.isTrue(businessName.length() <= 50, "Ime obrta može imati najviše 50 znakova, Vi ste napisali " + businessName.length() + " znakova.");
        businessRepository.findByBusinessName(businessName).ifPresent(business -> {
            throw new RequestDeniedException("Ime obrta '" + businessName + "' već postoji.");
        });

        //UPDATE
        Business business = businessRepository.findById(businessId).orElseThrow(
                () -> new EntityMissingException(Business.class, businessId));
        business.setBusinessName(businessName);
        return businessRepository.save(business);
    }

    @Override
    public Business updateBusinessDescription(long businessId, String businessDescription) {
        //CHECK DATA
        Assert.notNull(businessDescription, "Opis obrta mora biti predan.");
        Assert.hasText(businessDescription, "Opis obrta mora biti postavljen.");
        Assert.isTrue(businessDescription.length() <= 200, "Opis obrta može sadržavati najviše 200 znakova, Vi ste napisali " + businessDescription.length() + " znakova.");

        //UPDATE
        Business business = businessRepository.findById(businessId).orElseThrow(
                () -> new EntityMissingException(Business.class, businessId));
        business.setBusinessDescription(businessDescription);
        return businessRepository.save(business);
    }

    @Override
    public void deleteBusiness(String businessOIB) {
        Optional<Business> business = businessRepository.findByBusinessOIB(businessOIB);
        businessRepository.deleteById(business.get().getId());
    }

    @Override
    public void checkBusinessData (String businessName, String businessAddress, String businessCity, String businessOIB, String businessMobileNumber, String businessDescription){
        //BUSINESS NAME
        Assert.notNull(businessName, "Ime obrta mora biti predano.");
        Assert.hasText(businessName, "Ime obrta mora biti postavljeno.");
        Assert.isTrue(businessName.length() <= 50, "Ime obrta može imati najviše 50 znakova, Vi ste napisali " + businessName.length() + " znakova.");
        businessRepository.findByBusinessName(businessName).ifPresent(business -> {
            throw new RequestDeniedException("Ime obrta '" + businessName + "' već postoji.");
        });

        //BUSINESS ADDRESS
        Assert.notNull(businessAddress, "Adresa obrta mora biti predana.");
        Assert.hasText(businessAddress, "Adresa obrta mora biti postavljena.");

        //BUSINESS CITY
        Assert.notNull(businessCity, "Grad u kojem se nalazi obrt mora biti predan.");
        Assert.hasText(businessCity, "Grad u kojem se nalazi obrt mora biti postavljen.");

        //BUSINESS OIB
        Assert.notNull(businessOIB, "OIB obrta mora biti predan.");
        Assert.hasText(businessOIB, "OIB obrta mora biti postavljen.");
        Assert.isTrue(businessOIB.matches(businessOIBFormat), "OIB mora sadržavati točno 11 znamenki.");
        businessRepository.findByBusinessOIB(businessOIB).ifPresent(business -> {
            throw new RequestDeniedException("Već postoji obrt s OIB-om '" + businessOIB + "'");
        });

        //BUSINESS MOBILE NUMBER
        Assert.notNull(businessMobileNumber, "Kontakt broj obrta mora biti predan.");
        Assert.hasText(businessMobileNumber, "Kontakt broj obrta mora biti postavljen.");
        if(!(businessPhoneNumberFormat).matcher(businessMobileNumber).find()){
            throw new RequestDeniedException("Neispravan kontakt broj.");
        }
        //BUSINESS DESCRIPTION
        Assert.notNull(businessDescription, "Opis obrta mora biti predan.");
        Assert.hasText(businessDescription, "Opis obrta mora biti postavljen.");
        Assert.isTrue(businessDescription.length() <= 200, "Opis obrta može sadržavati najviše 200 znakova, Vi ste napisali " + businessDescription.length() + " znakova.");
    }

    @Override
    public Business setPromoteDuration(String businessOIB, String promoteDuration) {
        Business business = findByOIB(businessOIB).orElseThrow(
                () -> new EntityMissingException(Business.class, businessOIB)
        );

        business.setPromotionDuration(promoteDuration);
        business.setPromotionStart(LocalDate.now());
        return businessRepository.save(business);
    }

}
