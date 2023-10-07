package hr.fer.progi.simplicity.services.impl;

import hr.fer.progi.simplicity.repositories.CardRepository;
import hr.fer.progi.simplicity.services.CardService;
import hr.fer.progi.simplicity.entities.Card;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;

import javax.crypto.BadPaddingException;
import javax.crypto.Cipher;
import javax.crypto.IllegalBlockSizeException;
import javax.crypto.NoSuchPaddingException;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidAlgorithmParameterException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.spec.AlgorithmParameterSpec;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;
import java.util.Optional;
import java.util.regex.Pattern;

@Service
public class CardServiceJpa implements CardService {

    private static final Pattern DATE_PATTERN = Pattern.compile("^\\d{4}-\\d{2}-\\d{2}$");
    private static final String cardNumberFormat = "[0-9]{16}";
    private static final String cvvFormat = "[0-9]{3,4}";

    @Value("${app.encodingKey}")
    private String encodingKey;

    @Value("${app.ivVector}")
    private String ivVector;

    @Autowired
    private CardRepository cardRepository;

    @Override
    public Date checkCardData(String cardNumber, String expiryDateMonth, String expiryDateYear, String cvv) {
        //CARD NUMBER
        Assert.notNull(cardNumber, "Broj kartice mora biti predan.");
        Assert.hasText(cardNumber, "Broj kartice mora biti postavljen.");
        Assert.isTrue(cardNumber.matches(cardNumberFormat), "Neispravan broj kartice.");

        //CARD EXPIRY
        Assert.notNull(expiryDateMonth, "Mjesec isteka kartice mora biti predan.");
        Assert.notNull(expiryDateYear, "Godina isteka kartice mora biti predana.");
        Assert.hasText(expiryDateMonth, "Mjesec isteka kartice mora biti postavljen.");
        Assert.hasText(expiryDateYear, "Godina isteka kartice mora biti postavljena.");

        String dateStr = "" + expiryDateYear + "-" + expiryDateMonth + "-01";
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date pomDate = null;
        try{
            pomDate = dateFormat.parse(dateStr);
        }catch (ParseException exc){
            System.out.println("Greška u parsiranju datuma.");
        }

        Assert.isTrue(!DATE_PATTERN.matcher(pomDate.toString()).matches(), "Krivi format datuma, treba biti YYYY-MM-DD: ");
        Date currentDate = new Date();
        Assert.isTrue(pomDate.getTime() > currentDate.getTime(), "Kartica je istekla.");

        //CARD CVV
        Assert.notNull(cvv, "CVV broj kartice mora biti predan.");
        Assert.hasText(cvv, "CVV kartice mora biti postavljen..");
        Assert.isTrue(cvv.matches(cvvFormat), "CVV mora sadržavati 3 ili 4 znamenke.");

        return pomDate;
    }

    @Override
    public Card createNewCard(String cardNumber, Date endDate, String cvv) {
        try {
            String cardNumberEncrypted = encrypt(cardNumber, encodingKey, ivVector);
            return cardRepository.save(new Card(cardNumberEncrypted, endDate, cvv));
        } catch (NoSuchPaddingException e) {
            throw new RuntimeException(e);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        } catch (InvalidAlgorithmParameterException e) {
            throw new RuntimeException(e);
        } catch (InvalidKeyException e) {
            throw new RuntimeException(e);
        } catch (IllegalBlockSizeException e) {
            throw new RuntimeException(e);
        } catch (BadPaddingException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String encrypt(String input, String keyStr, String ivVectorStr) throws NoSuchPaddingException,
            NoSuchAlgorithmException, InvalidAlgorithmParameterException, InvalidKeyException, IllegalBlockSizeException, BadPaddingException {
        byte[] key = Base64.getDecoder().decode(keyStr);
        byte[] ivVector = Base64.getDecoder().decode(ivVectorStr);

        AlgorithmParameterSpec paramSpec = new IvParameterSpec(ivVector);
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"), paramSpec);

        byte[] outputBuffer = cipher.doFinal(input.getBytes());
        String ecnryptedOutput = Base64.getEncoder().encodeToString(outputBuffer);

        return ecnryptedOutput;
    }

    @Override
    public String decrypt(String input, String keyStr, String ivVectorStr) throws NoSuchPaddingException,
            NoSuchAlgorithmException, InvalidAlgorithmParameterException, InvalidKeyException, IllegalBlockSizeException, BadPaddingException {
        byte[] key = Base64.getDecoder().decode(keyStr);
        byte[] ivVector = Base64.getDecoder().decode(ivVectorStr);

        AlgorithmParameterSpec paramSpec = new IvParameterSpec(ivVector);
        Cipher cipher = Cipher.getInstance("AES/CBC/PKCS5Padding");
        cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"), paramSpec);

        byte[] outputBuffer = cipher.doFinal(Base64.getDecoder().decode(input));
        String decryptedOutput = new String(outputBuffer);

        return decryptedOutput;
    }

    @Override
    public Card deleteCard(String cardNumber) {
        Optional<Card> cardDB = cardRepository.findByCardNumber(cardNumber);
        cardRepository.deleteById(cardDB.get().getId());

        return cardDB.get();
    }

    @Override
    public Optional<Card> getCardById(long id) {
        return cardRepository.findById(id);
    }
}
