import React, { useState } from "react";
import { Dimensions, Image, Text, View } from "react-native";
import Carousel from "react-native-reanimated-carousel";
import { styles } from "./Styles";

const { width } = Dimensions.get("window");

export default function HomeCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0);

  const images = [
    {
      id: "1",
      image: require("../../../assets/images/img1.jpg"),
      title: "Descubra como a Plantia ajuda mais de 10.000 agricultores",
      body: "Faça parte de uma comunidade global que melhora a agricultura com tecnologia inteligente.",
    },
    {
      id: "2",
      image: require("../../../assets/images/img2.jpg"),
      title: "Proteja suas plantas com facilidade",
      body: "Mantenha seu jardim saudável com monitoramento e alertas baseados em IA.",
    },
    {
      id: "3",
      image: require("../../../assets/images/img3.jpg"),
      title: "Receba dicas de especialistas",
      body: "Aprenda as melhores práticas e truques compartilhados por agricultores experientes.",
    },
    {
      id: "4",
      image: require("../../../assets/images/img4.jpeg"),
      title: "Acompanhe o crescimento das suas plantas",
      body: "Monitore o desenvolvimento com o painel inteligente da Plantia.",
    },
  ];

  return (
    <View
      style={{
        height: "auto",
      }}
    >
      <Carousel
        loop
        autoPlay
        autoPlayInterval={4000}
        width={width}
        height={250}
        data={images}
        scrollAnimationDuration={800}
        onSnapToItem={(index) => setCurrentIndex(index)}
        renderItem={({ item }) => (
          <View style={styles.card}>
            <Image
              source={item.image}
              style={styles.image}
              resizeMode="cover"
            />

            <View style={styles.overlayContainer}>
              <Text style={styles.imageTitle}>{item.title}</Text>
              <Text style={styles.imageBody}>{item.body}</Text>
            </View>
          </View>
        )}
      />

      <View style={styles.dotsContainer}>
        {images.map((_, index) => (
          <View
            key={index}
            style={[styles.dot, currentIndex === index && styles.activeDot]}
          />
        ))}
      </View>
    </View>
  );
}
