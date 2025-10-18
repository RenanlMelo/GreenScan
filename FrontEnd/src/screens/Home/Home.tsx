import React from "react";
import { ScrollView, View } from "react-native";
import { styles } from "./Styles";
import HomeCarousel from "../../components/carousel/Carousel";
import ScanButton from "../../components/scanButton/ScanButton";
import { RecentHistory } from "../../components/recentHistory/RecentHistory";

export default function Home() {
  return (
    <ScrollView style={styles.container}>
      <HomeCarousel />
      <ScanButton />
      <RecentHistory />
    </ScrollView>
  );
}
